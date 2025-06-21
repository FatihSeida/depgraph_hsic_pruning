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

    def _l1_norm_plan(self, ratio: float) -> None:
        """Generate pruning plan based on convolution weight L1 norms."""
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")

        ch_scores: Dict[nn.Module, torch.Tensor] = {}
        for layer in self.layers:
            ch_scores[layer] = layer.weight.data.abs().sum(dim=(1, 2, 3))

        index_map = {(layer, idx): ch_scores[layer][idx].item() for layer in self.layers for idx in range(layer.out_channels)}
        groups = self.DG.get_all_groups(root_module_types=(nn.Conv2d,))
        scored_groups: List[Tuple[float, Any]] = []
        for g in groups:
            vals = []
            for dep, chs in g:
                mod = getattr(dep, "target", dep).module if hasattr(dep, "target") else dep.module
                if isinstance(mod, nn.Conv2d) and mod in ch_scores:
                    vals.extend(index_map.get((mod, c)) for c in chs if (mod, c) in index_map)
            if vals:
                scored_groups.append((sum(vals) / len(vals), g))

        scored_groups.sort(key=lambda x: x[0])

        remaining = {layer: layer.out_channels for layer in self.layers}
        num_to_prune = max(1, int(len(scored_groups) * ratio)) if ratio > 0 else 0
        plan: List[Any] = []
        for score, g in scored_groups:
            if len(plan) >= num_to_prune:
                break
            valid = True
            for dep, chs in g:
                mod = getattr(dep, "target", dep).module if hasattr(dep, "target") else dep.module
                if isinstance(mod, nn.Conv2d) and mod in remaining:
                    if remaining[mod] - len(chs) < 1:
                        valid = False
                        break
            if not valid:
                continue
            plan.append(g)
            for dep, chs in g:
                mod = getattr(dep, "target", dep).module if hasattr(dep, "target") else dep.module
                if isinstance(mod, nn.Conv2d) and mod in remaining:
                    remaining[mod] -= len(chs)

        self.pruning_plan = plan

    # ------------------------------------------------------------------
    # HSIC helpers
    # ------------------------------------------------------------------
    def _rbf_kernel(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        X = X.view(B, -1)
        dist = torch.cdist(X, X)
        K = torch.exp(-self.gamma * dist ** 2)
        H = torch.eye(B, device=K.device) - 1.0 / B
        return H @ K @ H

    def _hsic_scores(self, F: torch.Tensor, y: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        if F.shape[0] != y.shape[0]:
            self.logger.error(
                "Activation/label mismatch%s: %d activations vs %d labels",
                f" for layer {layer_idx}" if layer_idx is not None else "",
                F.shape[0],
                y.shape[0],
            )
            self.logger.debug("F.shape: %s, y.shape: %s", tuple(F.shape), tuple(y.shape))
            raise RuntimeError(
                "Mismatched number of activations and labels. Labels must be "
                "recorded for every forward pass using add_labels()."
            )
        B, C, H, W = F.shape
        Ky = self._rbf_kernel(y.unsqueeze(1))
        scores = []
        for j in range(C):
            Kj = self._rbf_kernel(F[:, j, :, :])
            scores.append((Kj * Ky).mean())
        return torch.stack(scores)


    def _log_dependency_status(self) -> None:
        """Report which layers were mapped to pruners in the current graph."""
        if self.DG is None or not hasattr(self.DG, "get_pruner_of_module"):
            return
        mapped = []
        for layer, name in zip(self.layers, self.layer_names):
            if self.DG.get_pruner_of_module(layer) is not None:
                mapped.append(name)
            else:
                self.logger.debug("Layer %s missing from dependency graph", name)
        if mapped:
            self.logger.info("Layers mapped to pruners: %s", mapped)
        else:
            self.logger.warning("No convolution layers mapped to pruners")

    # ------------------------------------------------------------------
    # BasePruningMethod interface
    # ------------------------------------------------------------------
    def analyze_model(self) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Analyzing model")
        import torch_pruning as tp

        self.logger.debug("Building dependency graph")
        try:
            device = next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover - model without parameters
            device = torch.device("cpu")
        if torch.is_tensor(self.example_inputs):
            self.example_inputs = self.example_inputs.to(device)
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
        self._dg_model = self.model
        self.logger.debug("Dependency graph built")
        if hasattr(self.DG, "get_pruning_group") and not hasattr(self.DG, "_wrapped"):
            orig = self.DG.get_pruning_group

            def _wrapped_get_pg(*args, **kwargs):
                attempts = 0
                while True:
                    try:
                        return orig(*args, **kwargs)
                    except Exception:
                        attempts += 1
                        if attempts >= 3:
                            raise

            self.DG.get_pruning_group = _wrapped_get_pg  # type: ignore[attr-defined]
            self.DG._wrapped = True  # type: ignore[attr-defined]
        try:
            param = next(self.model.parameters())
            model_device = param.device
            model_dtype = param.dtype
        except StopIteration:  # pragma: no cover - model without parameters
            model_device = torch.device("cpu")
            model_dtype = torch.float32
        example = self._inputs_tuple()[0]
        self.logger.info(
            "Model device: %s dtype: %s | Example input device: %s dtype: %s",
            model_device,
            model_dtype,
            example.device,
            example.dtype,
        )
        self.register_hooks()
        if self.layer_names:
            self.logger.info("Convolution layers found: %s", self.layer_names)

        missing: List[str] = []
        for layer, name in zip(self.layers, self.layer_names):
            pruner = self.DG.get_pruner_of_module(layer)
            if pruner is None:
                missing.append(name)

        if missing:
            self.logger.warning("Dependency graph missing layers: %s", missing)
        else:
            self.logger.info(
                "Successfully mapped %d layers to pruners", len(self.layers)
            )
        self.reset_records()

    def refresh_dependency_graph(self) -> None:  # pragma: no cover - heavy dependency
        """Rebuild the dependency graph without clearing recorded activations."""
        self.logger.info("Refreshing dependency graph")
        import torch_pruning as tp

        try:
            device = next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover - model without parameters
            device = torch.device("cpu")
        if torch.is_tensor(self.example_inputs):
            self.example_inputs = self.example_inputs.to(device)
        saved = (
            self.activations,
            self.layer_shapes,
            self.num_activations,
            self.labels,
        )
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
        self._dg_model = self.model
        self.register_hooks()
        (
            self.activations,
            self.layer_shapes,
            self.num_activations,
            self.labels,
        ) = saved

    def generate_pruning_mask(self, ratio: float, dataloader: Any | None = None) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        if dataloader is not None:
            self.reset_records()
            self._collect_activations(dataloader)
        if not self.activations or not self.labels:
            self.logger.warning(
                "No activations/labels collected. Falling back to L1-norm importance"
            )
            self._l1_norm_plan(ratio)
            return
        label_batches = len(self.labels)
        self.logger.info("Recorded %d label batches", label_batches)
        mismatch = False
        for idx, count in self.num_activations.items():
            self.logger.info("Layer %d recorded %d activations", idx, count)
            if count != label_batches:
                mismatch = True
                self.logger.warning(
                    "Layer %d has %d activations but %d labels", idx, count, label_batches
                )
        if label_batches < 2 and not mismatch:
            self.logger.warning(
                "Fewer than two label batches recorded; HSIC computation may be invalid"
            )
            self.logger.warning("Falling back to L1-norm importance")
            self._l1_norm_plan(ratio)
            return
        features = {}
        for idx, feats in self.activations.items():
            try:
                features[idx] = torch.cat(feats, dim=0)
            except RuntimeError:
                target_shape = self.layer_shapes.get(idx)
                pooled = [torch.nn.functional.adaptive_avg_pool2d(f, target_shape) if f.shape[2:] != target_shape else f for f in feats]
                features[idx] = torch.cat(pooled, dim=0)
        y = torch.cat(self.labels, dim=0)
        group_feats: List[torch.Tensor] = []
        hsic_values: List[torch.Tensor] = []
        group_info: List[Tuple[nn.Module, int]] = []
        for idx, layer in enumerate(self.layers):
            if idx not in features:
                continue
            F = features[idx]
            scores = self._hsic_scores(F, y, layer_idx=idx)
            for j in range(F.shape[1]):
                group_feats.append(F[:, j, :, :].mean(dim=(1, 2)))
                hsic_values.append(scores[j])
                group_info.append((layer, j))
        if not group_feats:
            raise RuntimeError("No feature activations recorded")
        X = torch.stack(group_feats, dim=1).cpu().numpy()
        y_np = y.view(len(y), -1).mean(dim=1).cpu().numpy()
        lasso = LassoLars(alpha=0.001)
        lasso.fit(X, y_np)
        coef = torch.tensor(lasso.coef_)
        importance = coef.abs() * torch.stack(hsic_values)
        index_map = {(layer, ch): i for i, (layer, ch) in enumerate(group_info)}
        groups = self.DG.get_all_groups(root_module_types=(nn.Conv2d,))
        scored_groups: List[Tuple[float, Any]] = []
        for g in groups:
            idxs = []
            for dep, chs in g:
                mod = getattr(dep, "target", dep).module if hasattr(dep, "target") else dep.module
                if isinstance(mod, nn.Conv2d) and mod in self.layers:
                    for c in chs:
                        key = (mod, c)
                        if key in index_map:
                            idxs.append(index_map[key])
            if idxs:
                score = importance[idxs].mean().item()
                scored_groups.append((score, g))
        scored_groups.sort(key=lambda x: x[0])
        num_to_prune = max(1, int(len(scored_groups) * ratio)) if ratio > 0 else 0
        remaining = {layer: layer.out_channels for layer in self.layers}
        plan: List[Any] = []
        for score, g in scored_groups:
            if len(plan) >= num_to_prune:
                break
            valid = True
            for dep, chs in g:
                mod = getattr(dep, "target", dep).module if hasattr(dep, "target") else dep.module
                if isinstance(mod, nn.Conv2d) and mod in remaining:
                    if remaining[mod] - len(chs) < 1:
                        valid = False
                        break
            if not valid:
                continue
            plan.append(g)
            for dep, chs in g:
                mod = getattr(dep, "target", dep).module if hasattr(dep, "target") else dep.module
                if isinstance(mod, nn.Conv2d) and mod in remaining:
                    remaining[mod] -= len(chs)
        self.pruning_plan = plan

    def apply_pruning(self, rebuild: bool = False) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Applying pruning")
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")
        import torch_pruning as tp

        model_changed = self.model is not self._dg_model
        if rebuild or model_changed:
            self.logger.debug("Rebuilding dependency graph before pruning")
            self.DG = tp.DependencyGraph()
            self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
            self._dg_model = self.model
            if not self.layers:
                self.register_hooks()
            self._log_dependency_status()

        try:
            for group in self.pruning_plan:
                try:
                    self.DG.prune_group(group)
                except AttributeError:
                    group.prune()
            try:
                tp.utils.remove_pruning_reparametrization(self.model)
            except Exception:  # pragma: no cover - safeguard against tp versions
                pass
        finally:
            self.remove_hooks()
