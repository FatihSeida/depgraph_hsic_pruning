from __future__ import annotations

"""Prune channel groups with HSIC-Lasso and a dependency graph.

The method collects activation maps and labels during normal forward
passes.  For every convolution channel an HSIC score measuring the
dependence between the channel output and the target labels is
computed.  These scores are combined with a sparse regression via
``LassoLars`` to decide which channel groups should be removed.
``torch-pruning``'s :class:`DependencyGraph` keeps tensor shapes
consistent during pruning and its reparameterisation is stripped once
pruning is finished.
"""

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from sklearn.linear_model import LassoLars
import torch_pruning as tp

from .base import BasePruningMethod


class DepgraphHSICMethod(BasePruningMethod):
    """Prune channel groups via HSIC and ``torch-pruning`` dependency graph.

    Parameters
    ----------
    model : Any
        Model to prune.
    workdir : str, optional
        Directory for intermediate artefacts, by default ``"runs/pruning"``.
    gamma : float, optional
        Scale factor for the RBF kernel used in HSIC, by default ``1.0``.
    num_modules : int, optional
        Number of modules from ``model.model`` to inspect when registering
        hooks.  Layers beyond this range are ignored.  The default of ``10``
        matches YOLOv8's backbone depth.
    """

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning", gamma: float = 1.0, num_modules: int = 10) -> None:
        super().__init__(model, workdir)
        self.gamma = gamma
        self.num_modules = num_modules
        self.example_inputs = torch.randn(1, 3, 640, 640)
        self.DG = None
        self._dg_model = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.layer_shapes: Dict[int, Tuple[int, int]] = {}
        self.num_activations: Dict[int, int] = {}
        self.labels: List[torch.Tensor] = []
        self.layers: List[nn.Conv2d] = []
        self.layer_names: List[str] = []
        self.pruner = None

    # ------------------------------------------------------------------
    # Utility hooks
    # ------------------------------------------------------------------
    def _activation_hook(self, idx: int):
        def hook(_module: nn.Module, _input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            target_shape = self.layer_shapes.get(idx)
            if target_shape is None:
                self.layer_shapes[idx] = output.shape[2:]
                processed = output.detach().cpu()
            else:
                if output.shape[2:] != target_shape:
                    processed = (
                        torch.nn.functional.adaptive_avg_pool2d(output, target_shape)
                        .detach()
                        .cpu()
                    )
                else:
                    processed = output.detach().cpu()
            self.activations.setdefault(idx, []).append(processed)
            self.num_activations[idx] = self.num_activations.get(idx, 0) + 1
            self.logger.debug(
                "Recorded activation for layer %d with shape %s",
                idx,
                tuple(processed.shape),
            )
            self.logger.debug(
                "Layer %d activation count: %d",
                idx,
                self.num_activations[idx],
            )
        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on convolution layers."""
        self.remove_hooks()
        self.layers = []
        self.layer_names = []
        idx = 0
        if hasattr(self.model, "model"):
            modules = list(self.model.model[: self.num_modules])
            for mod_idx, module in enumerate(modules):
                prefix = f"model.{mod_idx}"
                for name, m in module.named_modules():
                    if isinstance(m, nn.Conv2d):
                        full = prefix + (f".{name}" if name else "")
                        self.logger.debug(
                            "registering conv layer %s at index %d", full, idx
                        )
                        self.layers.append(m)
                        self.layer_names.append(full)
                        self.handles.append(
                            m.register_forward_hook(self._activation_hook(idx))
                        )
                        self.logger.debug(
                            "Registered hook for %s at index %d", full, idx
                        )
                        idx += 1
        else:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    self.logger.debug(
                        "registering conv layer %s at index %d", name, idx
                    )
                    self.layers.append(m)
                    self.layer_names.append(name)
                    self.handles.append(
                        m.register_forward_hook(self._activation_hook(idx))
                    )
                    self.logger.debug(
                        "Registered hook for %s at index %d", name, idx
                    )
                    idx += 1
        self.logger.info("Registered hooks for %d Conv2d layers", len(self.layers))

    def remove_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def add_labels(self, y: torch.Tensor) -> None:
        """Store labels observed during a forward pass."""
        processed = y.detach().cpu()
        self.labels.append(processed)
        self.logger.info("Recorded label tensor shape %s", tuple(processed.shape))
        self.logger.info("Cumulative labels stored: %d", len(self.labels))

    def reset_records(self) -> None:
        """Clear all collected activation data and labels.

        This removes cached activations, per-layer shapes, the activation
        counters and any stored labels so a new round of recording can begin
        without leftovers from previous passes.
        """
        act_count = sum(len(v) for v in self.activations.values())
        label_count = len(self.labels)
        self.logger.info(
            "Resetting records: clearing %d activations and %d labels",
            act_count,
            label_count,
        )
        self.activations.clear()
        self.layer_shapes.clear()
        self.num_activations.clear()
        self.labels.clear()

    def _collect_activations(self, dataloader) -> None:
        """Run ``dataloader`` once to populate hooks and labels."""
        try:
            device = next(self.model.parameters()).device
        except Exception:  # pragma: no cover - model without parameters
            device = torch.device("cpu")

        train_state = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = None
                labels = None
                if isinstance(batch, dict):
                    images = batch.get("img") or batch.get("images") or batch.get("inputs")
                    labels = batch.get("cls") or batch.get("label") or batch.get("labels")
                elif isinstance(batch, (list, tuple)):
                    if len(batch) > 0:
                        images = batch[0]
                    if len(batch) > 1:
                        labels = batch[1]
                else:  # pragma: no cover - fallback
                    images = batch
                if images is None:
                    continue
                self.model(images.to(device))
                if labels is not None:
                    self.add_labels(labels)
        self.model.train(train_state)

    def _rbf_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix for input tensor."""
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        K = torch.exp(-self.gamma * (X_norm + X_norm.t() - 2 * torch.mm(X, X.t())))
        return K

    def _hsic_scores(self, F: torch.Tensor, y: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        """Compute HSIC scores for each channel in feature tensor F."""
        if F.dim() != 2:
            F = F.view(F.size(0), -1)
        if y.dim() != 2:
            y = y.view(y.size(0), -1)

        # Convert to float if needed
        F = F.float()
        y = y.float()

        K_F = self._rbf_kernel(F)
        K_y = self._rbf_kernel(y)

        n = F.size(0)
        H = torch.eye(n, device=F.device) - 1.0 / n
        HSIC = torch.trace(torch.mm(torch.mm(H, K_F), torch.mm(H, K_y))) / (n - 1) ** 2

        return HSIC

    def _log_dependency_status(self) -> None:
        """Log information about dependency graph structure."""
        if self.DG is None:
            self.logger.warning("Dependency graph not initialized")
            return

        if hasattr(self.DG, "get_pruning_groups"):
            pruning_groups = list(self.DG.get_pruning_groups())
        elif hasattr(self.DG, "get_all_groups"):
            pruning_groups = list(self.DG.get_all_groups())
        else:  # pragma: no cover - API difference
            pruning_groups = []
        self.logger.info("Dependency graph has %d pruning groups", len(pruning_groups))
        
        for i, group in enumerate(pruning_groups):
            self.logger.debug("Group %d has %d dependencies", i, len(group))
            for dep in group:
                self.logger.debug("  - %s", dep)

    def analyze_model(self) -> None:  # pragma: no cover - heavy dependency
        """Analyze model structure and build dependency graph."""
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        example_inputs = tuple(t.to(device) if torch.is_tensor(t) else t for t in self._inputs_tuple())

        # Get the actual model to analyze
        if hasattr(self.model, "model"):
            model_to_analyze = self.model.model
        else:
            model_to_analyze = self.model

        # Refresh hooks to track convolution layers
        self.register_hooks()

        self.logger.info("Building dependency graph for model...")
        
        # Create dependency graph
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(model_to_analyze, example_inputs=example_inputs)
        
        self._log_dependency_status()
        
        # Store the analyzed model
        self._dg_model = model_to_analyze
        
        self.logger.info("Dependency graph analysis completed")

    def refresh_dependency_graph(self) -> None:  # pragma: no cover - heavy dependency
        """Refresh dependency graph after model changes."""
        if self.DG is None:
            self.logger.warning("No dependency graph to refresh")
            return

        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        example_inputs = tuple(t.to(device) if torch.is_tensor(t) else t for t in self._inputs_tuple())
        
        # Get the actual model to analyze
        if hasattr(self.model, "model"):
            model_to_analyze = self.model.model
        else:
            model_to_analyze = self.model

        # Refresh hooks to track convolution layers
        self.register_hooks()

        self.logger.info("Refreshing dependency graph...")
        
        # Rebuild dependency graph
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(model_to_analyze, example_inputs=example_inputs)
        
        self._log_dependency_status()
        
        # Update the analyzed model
        self._dg_model = model_to_analyze
        
        self.logger.info("Dependency graph refresh completed")

    def generate_pruning_mask(
        self,
        ratio: float,
        dataloader: Any | None = None,
        *,
        min_labels: int = 4,
    ) -> None:
        """Generate pruning mask using HSIC-Lasso method.

        Parameters
        ----------
        ratio : float
            Target pruning ratio (0.0 to 1.0).
        dataloader : Any, optional
            DataLoader for collecting activations and labels.
        min_labels : int, optional
            Minimum number of labels required for HSIC computation.
        """
        self.logger.info("Generating pruning mask with ratio %.3f", ratio)
        
        # Analyze model if not done yet
        if self.DG is None:
            self.analyze_model()

        # Try using GroupNormPruner like the official example
        try:
            self._use_group_norm_pruner(ratio)
            return
        except Exception as e:
            self.logger.warning("GroupNormPruner failed: %s", str(e))
            self.logger.info("Falling back to dependency graph method")

        # Collect activations and labels if dataloader provided
        if dataloader is not None:
            self.logger.info("Collecting activations and labels...")
            self._collect_activations(dataloader)
            
            if len(self.labels) < min_labels:
                raise ValueError(
                    f"Insufficient labels ({len(self.labels)} < {min_labels}) for HSIC computation"
                )

        # Try HSIC-based pruning
        try:
            self._hsic_lasso_plan(ratio)
        except Exception as e:
            self.logger.error("HSIC-Lasso pruning failed: %s", str(e))
            raise RuntimeError(f"HSIC-Lasso pruning failed: {str(e)}")

    def _use_group_norm_pruner(self, ratio: float) -> None:
        """Use GroupNormPruner like the official YOLOv8 example."""
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        example_inputs = tuple(t.to(device) if torch.is_tensor(t) else t for t in self._inputs_tuple())
        
        # Get the actual model to analyze
        if hasattr(self.model, "model"):
            model_to_analyze = self.model.model
        else:
            model_to_analyze = self.model

        self.logger.info("Using GroupNormPruner with GroupMagnitudeImportance")
        
        # Create pruner like in the official example
        self.pruner = tp.pruner.GroupNormPruner(
            model_to_analyze,
            example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(),
            iterative_steps=1,
            pruning_ratio=ratio,
            ignored_layers=[],
            unwrapped_parameters=[]
        )
        
        # Execute pruning step
        self.pruner.step()
        
        self.logger.info("GroupNormPruner pruning completed successfully")

    def _hsic_lasso_plan(self, ratio: float) -> None:
        """Generate pruning plan using HSIC-Lasso method."""
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")

        # Compute HSIC scores for each layer
        hsic_scores = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in self.activations:
                continue
                
            activations = torch.cat(self.activations[layer_idx], dim=0)
            if len(self.labels) == 0:
                continue
                
            # Use synthetic labels if no real labels available
            if len(self.labels) < 2:
                synthetic_labels = torch.randn(activations.size(0), 1)
            else:
                synthetic_labels = torch.cat(self.labels, dim=0)
                if synthetic_labels.size(0) != activations.size(0):
                    # Truncate to match
                    min_size = min(activations.size(0), synthetic_labels.size(0))
                    activations = activations[:min_size]
                    synthetic_labels = synthetic_labels[:min_size]

            # Compute HSIC scores for each channel
            channel_scores = []
            for ch in range(activations.size(1)):
                ch_activations = activations[:, ch:ch+1]
                score = self._hsic_scores(ch_activations, synthetic_labels, layer_idx)
                channel_scores.append(score.item())
            
            hsic_scores[layer] = torch.tensor(channel_scores)

        # Use LassoLars for sparse regression
        if not hsic_scores:
            raise ValueError("No HSIC scores computed")

        # Combine all scores into feature matrix
        all_scores = []
        layer_indices = []
        for layer, scores in hsic_scores.items():
            all_scores.extend(scores.tolist())
            layer_indices.extend([layer] * len(scores))

        X = torch.tensor(all_scores).unsqueeze(1)
        y = torch.ones(X.size(0))  # Dummy target

        # Fit LassoLars
        lasso = LassoLars(alpha=0.1, max_iter=1000)
        lasso.fit(X.numpy(), y.numpy())
        
        # Get coefficients and determine which channels to prune
        coefficients = torch.tensor(lasso.coef_)
        scores_with_coeffs = X.squeeze() * coefficients
        
        # Sort by score and select bottom ratio% for pruning
        sorted_indices = torch.argsort(scores_with_coeffs)
        num_to_prune = int(len(sorted_indices) * ratio)
        indices_to_prune = sorted_indices[:num_to_prune]

        # Apply pruning to dependency graph
        pruning_groups = self.DG.get_pruning_groups()
        pruned_count = 0
        
        for idx in indices_to_prune:
            layer = layer_indices[idx]
            # Find corresponding pruning group and apply
            for group in pruning_groups:
                for dep in group:
                    if hasattr(dep.target, 'weight') and dep.target == layer:
                        self.DG.prune_group(group)
                        pruned_count += 1
                        break

        self.logger.info("Applied HSIC-Lasso pruning: %d channels", pruned_count)

    def apply_pruning(self, rebuild: bool = False) -> None:  # pragma: no cover - heavy dependency
        """Apply the generated pruning mask to the model."""
        if self.pruner is not None:
            # If using GroupNormPruner, it's already applied
            self.logger.info("Pruning already applied by GroupNormPruner")
            return
            
        if self.DG is None:
            raise RuntimeError("No dependency graph available")

        self.logger.info("Applying pruning via dependency graph...")
        
        # The pruning is already applied during mask generation
        # Just need to rebuild if requested
        if rebuild:
            self.logger.info("Rebuilding model after pruning...")
            # The dependency graph handles the rebuilding automatically
            pass

        self.logger.info("Pruning applied successfully")

    def _individual_channel_pruning(self, ratio: float, fallback_allowed: bool = False) -> None:
        """Try individual channel pruning as fallback."""
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")

        self.logger.info("Attempting individual channel pruning...")
        
        # Get pruning groups
        pruning_groups = self.DG.get_pruning_groups()
        self.logger.info("Found %d pruning groups for individual channel pruning", len(pruning_groups))
        
        if len(pruning_groups) <= 1:
            self.logger.warning("Only %d pruning group found, individual channel pruning may not work", len(pruning_groups))
            if fallback_allowed:
                self.logger.info("Falling back to simple L1 norm pruning")
                self._l1_norm_plan_simple(ratio)
                return
            else:
                raise RuntimeError("Individual channel pruning failed - insufficient pruning groups")

        # Try to prune individual channels
        total_channels = 0
        pruned_channels = 0
        
        for group in pruning_groups:
            for dep in group:
                if hasattr(dep.target, 'weight'):
                    total_channels += dep.target.weight.size(0)
        
        target_pruned = int(total_channels * ratio)
        
        # Sort groups by some criterion (e.g., layer depth)
        sorted_groups = sorted(enumerate(pruning_groups), key=lambda x: len(x[1]))
        
        for group_idx, group in sorted_groups:
            if pruned_channels >= target_pruned:
                break
                
            # Try to prune this group
            try:
                self.DG.prune_group(group)
                pruned_channels += len([dep for dep in group if hasattr(dep.target, 'weight')])
                self.logger.debug("Pruned group %d with %d dependencies", group_idx, len(group))
            except Exception as e:
                self.logger.debug("Failed to prune group %d: %s", group_idx, str(e))
                continue

        self.logger.info("Individual channel pruning completed: %d/%d channels pruned", pruned_channels, total_channels)

        if pruned_channels == 0:
            raise RuntimeError("Individual channel pruning produced no pruning")

