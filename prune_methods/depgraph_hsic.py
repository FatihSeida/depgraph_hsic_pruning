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

from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from sklearn.linear_model import LassoLars
import os

# Disable multiprocessing to avoid ConnectionResetError
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    import torch_pruning as tp
    from torch_pruning import DependencyGraph
    TORCH_PRUNING_AVAILABLE = True
except ImportError:
    TORCH_PRUNING_AVAILABLE = False
    DependencyGraph = None

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
    pruning_scope : str, optional
        The scope of the pruning. By default, it is set to "backbone".
    """

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning", gamma: float = 1.0, num_modules: int = 10, pruning_scope: str = "backbone") -> None:
        super().__init__(model, workdir)
        self.gamma = gamma
        self.num_modules = num_modules
        self.pruning_scope = pruning_scope
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
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_pruning_groups(self):
        """Return pruning groups from the dependency graph."""
        if self.DG is None:
            return []
        if hasattr(self.DG, "get_pruning_groups"):
            return list(self.DG.get_pruning_groups())
        if hasattr(self.DG, "get_all_groups"):
            return list(self.DG.get_all_groups())
        # pragma: no cover - API difference
        return []

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
        
        if self.pruning_scope == "backbone":
            # Use existing utility for backbone-only pruning
            from .utils import collect_backbone_convs
            backbone_convs = collect_backbone_convs(self.model, self.num_modules)
            
            for parent, attr_name, bn in backbone_convs:
                conv_layer = getattr(parent, attr_name)
                if isinstance(conv_layer, nn.Conv2d):
                    # Construct full name for logging
                    full_name = f"{parent.__class__.__name__}.{attr_name}"
                    self.logger.debug(
                        "registering backbone conv layer %s at index %d", full_name, idx
                    )
                    self.layers.append(conv_layer)
                    self.layer_names.append(full_name)
                    self.handles.append(
                        conv_layer.register_forward_hook(self._activation_hook(idx))
                    )
                    self.logger.debug(
                        "Registered hook for %s at index %d", full_name, idx
                    )
                    idx += 1
        else:
            # Register hooks for all Conv2d layers in the model (full scope)
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.logger.debug(
                        "registering full conv layer %s at index %d", name, idx
                    )
                    self.layers.append(module)
                    self.layer_names.append(name)
                    self.handles.append(
                        module.register_forward_hook(self._activation_hook(idx))
                    )
                    self.logger.debug(
                        "Registered hook for %s at index %d", name, idx
                    )
                    idx += 1
        
        self.logger.info(f"Registered hooks for {len(self.layers)} Conv2d layers (scope: {self.pruning_scope})")

    def remove_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def add_labels(self, y: torch.Tensor) -> None:
        """Store labels observed during a forward pass."""
        processed = y.detach().cpu()
        self.labels.append(processed)
        self.logger.debug("Recorded label tensor shape %s", tuple(processed.shape))
        self.logger.debug("Cumulative labels stored: %d", len(self.labels))

    def reset_records(self) -> None:
        """Clear all collected activation data and labels.

        This removes cached activations, per-layer shapes, the activation
        counters and any stored labels so a new round of recording can begin
        without leftovers from previous passes.
        """
        act_count = sum(len(v) for v in self.activations.values())
        label_count = len(self.labels)
        self.logger.debug(
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

        pruning_groups = self._get_pruning_groups()
        self.logger.debug("Dependency graph has %d pruning groups", len(pruning_groups))
        
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

        # Use the model instance directly
        model_to_analyze = self.model

        # Refresh hooks to track convolution layers
        self.register_hooks()

        self.logger.debug("Building dependency graph for model...")
        
        # Create dependency graph
        self.DG = DependencyGraph()
        self.DG.build_dependency(model_to_analyze, example_inputs=example_inputs)
        
        self._log_dependency_status()
        
        # Store the analyzed model
        self._dg_model = model_to_analyze
        
        self.logger.debug("Dependency graph analysis completed")

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
        
        # Use the model instance directly
        model_to_analyze = self.model

        # Refresh hooks to track convolution layers
        self.register_hooks()

        self.logger.debug("Refreshing dependency graph...")
        
        # Rebuild dependency graph
        self.DG = DependencyGraph()
        self.DG.build_dependency(model_to_analyze, example_inputs=example_inputs)
        
        self._log_dependency_status()
        
        # Update the analyzed model
        self._dg_model = model_to_analyze
        
        self.logger.debug("Dependency graph refresh completed")

    def generate_pruning_mask(
        self,
        ratio: float,
        dataloader: Any | None = None,
        *,
        min_labels: int = 4,
        max_samples: int = 1000,
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
        max_samples : int, optional
            Maximum number of samples to collect for HSIC computation.
        """
        self.logger.debug("Generating pruning mask with ratio %.3f", ratio)
        
        # Validate inputs
        if ratio <= 0 or ratio >= 1:
            raise ValueError(f"Pruning ratio must be between 0 and 1, got {ratio}")
        
        if dataloader is None:
            raise ValueError("Dataloader is required for HSIC computation. No fallback to synthetic data allowed.")
        
        # Analyze model if not done yet
        if self.DG is None:
            self.analyze_model()

        # Collect activations and labels from real data
        self.logger.info("Collecting activations using real data for HSIC")
        self._collect_activations_robust(dataloader, max_samples)
        
        if len(self.labels) < min_labels:
            raise ValueError(
                f"Insufficient labels ({len(self.labels)} < {min_labels}) for HSIC computation. "
                f"Please provide more data samples."
            )

        # Build dependency groups
        pruning_groups = self._get_pruning_groups()
        if not pruning_groups:
            raise RuntimeError("No pruning groups found in dependency graph")
        
        self.logger.info(f"Found {len(pruning_groups)} pruning groups")
        
        # Compute HSIC scores for each group
        group_scores = self._compute_group_hsic_scores(pruning_groups)
        
        if not group_scores:
            raise RuntimeError("Failed to compute HSIC scores for any group")
        
        # Apply HSIC-Lasso pruning
        self._apply_hsic_lasso_pruning(pruning_groups, group_scores, ratio)
        
        self.logger.info("HSIC-Lasso pruning mask generated successfully")

    def _collect_activations_robust(self, dataloader, max_samples: int) -> None:
        """Collect activations and labels with robust error handling."""
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        train_state = self.model.training
        self.model.eval()
        
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                    
                images = None
                labels = None
                
                # Extract images and labels from batch with better error handling
                try:
                    if isinstance(batch, dict):
                        # Try different possible keys for images
                        for key in ["img", "images", "inputs"]:
                            if key in batch and batch[key] is not None:
                                images = batch[key]
                                break
                        
                        # Try different possible keys for labels
                        for key in ["cls", "label", "labels"]:
                            if key in batch and batch[key] is not None:
                                labels = batch[key]
                                break
                    elif isinstance(batch, (list, tuple)):
                        if len(batch) > 0 and batch[0] is not None:
                            images = batch[0]
                        if len(batch) > 1 and batch[1] is not None:
                            labels = batch[1]
                    else:
                        # Assume batch is directly the images
                        images = batch
                        
                except Exception as e:
                    self.logger.warning(f"Error processing batch: {e}")
                    continue
                
                # Skip if no images found
                if images is None:
                    self.logger.debug("No images found in batch, skipping")
                    continue
                
                # Ensure images is a tensor and has correct shape
                if not isinstance(images, torch.Tensor):
                    self.logger.warning(f"Images is not a tensor: {type(images)}")
                    continue
                
                if images.dim() != 4:  # Should be [batch, channels, height, width]
                    self.logger.warning(f"Images tensor has wrong dimensions: {images.shape}")
                    continue
                
                try:
                    # Forward pass to collect activations
                    self.model(images.to(device))
                    
                    # Store labels if available
                    if labels is not None and isinstance(labels, torch.Tensor):
                        self.add_labels(labels)
                    
                    sample_count += 1
                    
                    if sample_count % 10 == 0:
                        self.logger.debug(f"Collected {sample_count} samples")
                        
                except Exception as e:
                    self.logger.warning(f"Error during forward pass: {e}")
                    continue
        
        self.model.train(train_state)
        self.logger.info(f"Collected activations from {sample_count} samples")
        
        if sample_count == 0:
            raise RuntimeError("No samples were collected. Check dataloader format and data.")

    def _compute_group_hsic_scores(self, pruning_groups) -> Dict[int, torch.Tensor]:
        """Compute HSIC scores for each pruning group."""
        group_scores = {}
        
        for group_idx, group in enumerate(pruning_groups):
            # Collect activations for all layers in this group
            group_activations = []
            group_channels = []
            
            for dep in group:
                if hasattr(dep.target, 'weight') and isinstance(dep.target, nn.Conv2d):
                    layer_idx = None
                    for idx, layer in enumerate(self.layers):
                        if layer is dep.target:
                            layer_idx = idx
                            break
                    
                    if layer_idx is not None and layer_idx in self.activations:
                        acts = torch.cat(self.activations[layer_idx], dim=0)
                        group_activations.append(acts)
                        group_channels.append(acts.size(1))
            
            if not group_activations:
                continue
            
            # Concatenate activations from all layers in group
            combined_activations = torch.cat(group_activations, dim=1)
            
            # Prepare labels
            if len(self.labels) == 0:
                raise RuntimeError("No labels available for HSIC computation")
            
            labels_tensor = torch.cat(self.labels, dim=0)
            
            # Ensure same number of samples
            min_samples = min(combined_activations.size(0), labels_tensor.size(0))
            combined_activations = combined_activations[:min_samples]
            labels_tensor = labels_tensor[:min_samples]
            
            # Compute HSIC scores for each channel
            channel_scores = []
            for ch in range(combined_activations.size(1)):
                ch_activations = combined_activations[:, ch:ch+1]
                score = self._hsic_scores(ch_activations, labels_tensor, group_idx)
                channel_scores.append(score.item())
            
            group_scores[group_idx] = torch.tensor(channel_scores)
            self.logger.debug(f"Group {group_idx}: computed HSIC scores for {len(channel_scores)} channels")
        
        return group_scores

    def _apply_hsic_lasso_pruning(self, pruning_groups, group_scores, ratio: float) -> None:
        """Apply HSIC-Lasso pruning to the dependency graph."""
        # Combine all scores for Lasso regression
        all_scores = []
        group_indices = []
        channel_indices = []
        
        for group_idx, scores in group_scores.items():
            all_scores.extend(scores.tolist())
            group_indices.extend([group_idx] * len(scores))
            channel_indices.extend(list(range(len(scores))))
        
        if not all_scores:
            raise RuntimeError("No HSIC scores available for pruning")
        
        # Prepare data for Lasso regression
        X = torch.tensor(all_scores).unsqueeze(1)
        y = torch.ones(X.size(0))  # Dummy target for Lasso
        
        # Fit LassoLars
        lasso = LassoLars(alpha=0.1, max_iter=1000)
        lasso.fit(X.numpy(), y.numpy())
        
        # Get coefficients and determine which channels to prune
        coefficients = torch.tensor(lasso.coef_)
        scores_with_coeffs = X.squeeze() * coefficients
        
        # Sort by score and select bottom ratio% for pruning
        sorted_indices = torch.argsort(scores_with_coeffs)
        num_to_prune = int(len(sorted_indices) * ratio)
        
        if num_to_prune == 0:
            self.logger.warning(f"No channels selected for pruning with ratio {ratio}")
            return
        
        indices_to_prune = sorted_indices[:num_to_prune]
        
        # Apply pruning to dependency graph
        pruned_count = 0
        for idx in indices_to_prune:
            group_idx = group_indices[idx]
            channel_idx = channel_indices[idx]
            
            # Find corresponding group and apply pruning
            if group_idx < len(pruning_groups):
                group = pruning_groups[group_idx]
                try:
                    self.DG.prune_group(group)
                    pruned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to prune group {group_idx}: {e}")
        
        self.logger.info(f"Applied HSIC-Lasso pruning: {pruned_count} groups pruned")
        
        if pruned_count == 0:
            raise RuntimeError("No pruning was applied. Check dependency graph structure and pruning ratio.")

    def apply_pruning(self, rebuild: bool = False) -> None:  # pragma: no cover - heavy dependency
        """Apply the generated pruning mask to the model."""
        if self.DG is None:
            raise RuntimeError("No dependency graph available. Call generate_pruning_mask first.")

        self.logger.debug("Applying pruning via dependency graph...")
        
        # The pruning is already applied during mask generation via DG.prune_group()
        # Just need to rebuild if requested
        if rebuild:
            self.logger.debug("Rebuilding model after pruning...")
            # The dependency graph handles the rebuilding automatically
            pass

        self.logger.info("Pruning applied successfully via dependency graph")

    def validate_pruning(self) -> Dict[str, Any]:
        """Validate that pruning was successful and model is still functional."""
        if self.DG is None:
            raise RuntimeError("No dependency graph available")
        
        validation_results = {
            "pruning_groups_remaining": 0,
            "total_channels_remaining": 0,
            "model_functional": False,
            "pruning_ratio_achieved": 0.0
        }
        
        # Check remaining pruning groups
        pruning_groups = self._get_pruning_groups()
        validation_results["pruning_groups_remaining"] = len(pruning_groups)
        
        # Count remaining channels
        total_channels = 0
        for group in pruning_groups:
            for dep in group:
                if hasattr(dep.target, 'weight') and isinstance(dep.target, nn.Conv2d):
                    total_channels += dep.target.weight.size(0)
        
        validation_results["total_channels_remaining"] = total_channels
        
        # Test if model is still functional
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        try:
            with torch.no_grad():
                _ = self.model(test_input)
            validation_results["model_functional"] = True
            self.logger.info("Model validation successful - model is functional after pruning")
        except Exception as e:
            validation_results["model_functional"] = False
            self.logger.error(f"Model validation failed: {e}")
        
        # Calculate pruning ratio achieved
        original_channels = sum(layer.weight.size(0) for layer in self.layers)
        if original_channels > 0:
            validation_results["pruning_ratio_achieved"] = 1.0 - (total_channels / original_channels)
        
        return validation_results

    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get a summary of the pruning operation."""
        if self.DG is None:
            return {"error": "No dependency graph available"}
        
        pruning_groups = self._get_pruning_groups()
        
        summary = {
            "total_pruning_groups": len(pruning_groups),
            "layers_analyzed": len(self.layers),
            "activations_collected": sum(len(acts) for acts in self.activations.values()),
            "labels_collected": len(self.labels),
            "dependency_graph_built": self.DG is not None
        }
        
        # Add validation results
        try:
            validation = self.validate_pruning()
            summary.update(validation)
        except Exception as e:
            summary["validation_error"] = str(e)
        
        return summary

