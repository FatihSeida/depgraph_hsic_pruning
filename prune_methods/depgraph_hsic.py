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
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.layer_shapes: Dict[int, Tuple[int, int]] = {}
        self.num_activations: Dict[int, int] = {}
        self.labels: List[torch.Tensor] = []
        self.layers: List[nn.Conv2d] = []
        self.layer_names: List[str] = []
        self.adjacency: torch.Tensor | None = None
        self.channel_groups: List[List[Tuple[int, int]]] = []

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

    # ------------------------------------------------------------------
    # Adjacency and grouping helpers
    # ------------------------------------------------------------------
    def _build_adjacency(self) -> None:
        """Construct an adjacency matrix between convolution layers."""
        if self.DG is None:
            self.adjacency = None
            return
        n = len(self.layers)
        adj = torch.zeros(n, n, dtype=torch.int8)
        for group in self.DG.get_all_groups(root_module_types=(nn.Conv2d,)):
            conv_ids: List[int] = []
            for dep, _ in group:
                mod = dep.target.module
                if isinstance(mod, nn.Conv2d) and mod in self.layers:
                    conv_ids.append(self.layers.index(mod))
            for i in range(len(conv_ids)):
                for j in range(i + 1, len(conv_ids)):
                    adj[conv_ids[i], conv_ids[j]] = 1
                    adj[conv_ids[j], conv_ids[i]] = 1
        self.adjacency = adj

    def _build_channel_groups(self) -> None:
        """Group channels across layers via BFS on the adjacency matrix."""
        if self.adjacency is None or self.DG is None:
            self.channel_groups = []
            return
        import torch_pruning as tp
        visited = set()
        groups: List[List[Tuple[int, int]]] = []
        for li, layer in enumerate(self.layers):
            pruner = self.DG.get_pruner_of_module(layer)
            if pruner is None or pruner.get_out_channels(layer) is None:
                continue
            out_ch = pruner.get_out_channels(layer)
            for ci in range(out_ch):
                if (li, ci) in visited:
                    continue
                queue = [(li, ci)]
                current: List[Tuple[int, int]] = []
                while queue:
                    lidx, cidx = queue.pop(0)
                    if (lidx, cidx) in visited:
                        continue
                    visited.add((lidx, cidx))
                    current.append((lidx, cidx))
                    conv = self.layers[lidx]
                    try:
                        grp = self.DG.get_pruning_group(
                            conv, tp.prune_conv_out_channels, [cidx]
                        )
                    except ValueError:
                        continue
                    for dep, idxs in grp:
                        mod = dep.target.module
                        if isinstance(mod, nn.Conv2d) and mod in self.layers:
                            ni = self.layers.index(mod)
                            for ch in idxs:
                                if (ni, ch) not in visited:
                                    queue.append((ni, ch))
                if current:
                    groups.append(current)
        self.channel_groups = groups

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
        self.logger.debug("Dependency graph built")
        self.register_hooks()
        self._build_adjacency()
        self._build_channel_groups()
        self.reset_records()

    def generate_pruning_mask(self, ratio: float) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        if not self.activations or not self.labels:
            self.logger.debug(
                "generate_pruning_mask called with %d labels and activations for %d layers",
                len(self.labels),
                len(self.activations),
            )
            raise RuntimeError("No activations/labels collected. Run a forward pass first.")
        label_batches = len(self.labels)
        self.logger.info("Recorded %d label batches", label_batches)
        for idx, count in self.num_activations.items():
            self.logger.info("Layer %d recorded %d activations", idx, count)
            if count != label_batches:
                self.logger.warning(
                    "Layer %d has %d activations but %d labels", idx, count, label_batches
                )
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
        group_scores: List[Tuple[float, List[Tuple[nn.Module, int]]]] = []
        total_channels = len(importance)
        for g in self.channel_groups:
            idxs = []
            chans: List[Tuple[nn.Module, int]] = []
            for li, ci in g:
                key = (self.layers[li], ci)
                if key in index_map:
                    idxs.append(index_map[key])
                    chans.append(key)
            if idxs:
                score = importance[idxs].mean().item()
                group_scores.append((score, chans))

        group_scores.sort(key=lambda x: x[0])
        target = int(total_channels * ratio)
        removed = 0
        name_map = dict(zip(self.layers, self.layer_names))
        self.pruning_plan = {}
        for score, chans in group_scores:
            if removed >= target:
                break
            for layer, ch in chans:
                name = name_map.get(layer)
                if name is not None:
                    self.pruning_plan.setdefault(name, []).append(ch)
            removed += len(chans)

    def apply_pruning(self) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Applying pruning")
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")
        import torch_pruning as tp

        # Always rebuild the dependency graph in case the model changed
        self.logger.debug("Rebuilding dependency graph before pruning")
        self.DG = tp.DependencyGraph()
        try:
            self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
            saved = (
                self.activations,
                self.layer_shapes,
                self.num_activations,
                self.labels,
            )
            self.register_hooks()
            self.activations, self.layer_shapes, self.num_activations, self.labels = saved
        except Exception as build_err:
            self.logger.error("Dependency graph build failed: %s", build_err)
            saved = (
                self.activations,
                self.layer_shapes,
                self.num_activations,
                self.labels,
            )
            self.logger.info("Analyzing model to rebuild dependency graph")
            self.analyze_model()
            self.activations, self.layer_shapes, self.num_activations, self.labels = saved
            self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
            saved = (
                self.activations,
                self.layer_shapes,
                self.num_activations,
                self.labels,
            )
            self.register_hooks()
            self.activations, self.layer_shapes, self.num_activations, self.labels = saved

        named_modules = dict(self.model.named_modules())

        for name, idxs in self.pruning_plan.items():
            layer = named_modules.get(name)
            if layer is None:
                raise RuntimeError(f"Layer {name!r} not found in model")
            if layer not in self.layers:
                raise RuntimeError(
                    f"Layer {name!r} not found in active layer list. Run analyze_model() after changing layers."
                )
            unique = sorted(set(idxs))
            self.logger.debug("pruning %s channels %s", name, unique)
            self.logger.debug("Attempting to obtain pruning group for %s", name)
            try:
                group = self.DG.get_pruning_group(
                    layer,
                    tp.prune_conv_out_channels,
                    unique,
                )
            except ValueError as e:
                self.logger.debug("get_pruning_group failed: %s", e)
                self.logger.info("Rebuilding dependency graph before pruning")
                # recreate the DependencyGraph in case the model changed
                self.DG = tp.DependencyGraph()
                self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
                tmp = (
                    self.activations,
                    self.layer_shapes,
                    self.num_activations,
                    self.labels,
                )
                self.register_hooks()
                self.activations, self.layer_shapes, self.num_activations, self.labels = tmp
                named_modules = dict(self.model.named_modules())
                self.logger.debug(
                    "dependency graph modules: %s", list(named_modules.keys())
                )
                layer = named_modules.get(name)
                if layer is None:
                    raise RuntimeError(
                        "Layer %s not found after model update. "
                        "Run analyze_model() after changing layers." % name
                    )
                if layer not in self.layers:
                    raise RuntimeError(
                        "Layer %s not found in active layer list after rebuild. "
                        "Run analyze_model() after changing layers." % name
                    )
                self.logger.debug(
                    "Retrying get_pruning_group for %s with %s", name, unique
                )
                try:
                    group = self.DG.get_pruning_group(
                        layer,
                        tp.prune_conv_out_channels,
                        unique,
                    )
                except ValueError as err:
                    self.logger.error(
                        "get_pruning_group failed again for %s: %s", name, err
                    )
                    self.logger.debug(
                        "Analyzing model and rebuilding dependency graph for final retry"
                    )
                    saved = (
                        self.activations,
                        self.layer_shapes,
                        self.num_activations,
                        self.labels,
                    )
                    self.analyze_model()
                    self.activations, self.layer_shapes, self.num_activations, self.labels = saved
                    self.DG = tp.DependencyGraph()
                    self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())
                    temp = (
                        self.activations,
                        self.layer_shapes,
                        self.num_activations,
                        self.labels,
                    )
                    self.register_hooks()
                    self.activations, self.layer_shapes, self.num_activations, self.labels = temp
                    named_modules = dict(self.model.named_modules())
                    layer = named_modules.get(name)
                    if layer is None:
                        raise RuntimeError(
                            "Layer %s not found after model update. "
                            "Run analyze_model() after changing layers." % name
                        )
                    if layer not in self.layers:
                        raise RuntimeError(
                            "Layer %s not found in active layer list after rebuild. "
                            "Run analyze_model() after changing layers." % name
                        )
                    self.logger.debug(
                        "Final attempt get_pruning_group for %s with %s", name, unique
                    )
                    try:
                        group = self.DG.get_pruning_group(
                            layer,
                            tp.prune_conv_out_channels,
                            unique,
                        )
                    except ValueError as err2:
                        self.logger.error(
                            "get_pruning_group failed third time for %s: %s", name, err2
                        )
                        try:
                            model_device = next(self.model.parameters()).device
                        except StopIteration:  # pragma: no cover - model without parameters
                            model_device = torch.device("cpu")
                        inputs_device = (
                            self.example_inputs.device
                            if torch.is_tensor(self.example_inputs)
                            else None
                        )
                        self.logger.error(
                            "Model device: %s, example_inputs device: %s",
                            model_device,
                            inputs_device,
                        )
                        raise RuntimeError(
                            f"Failed to obtain pruning group for layer {name} after model update. "
                            "Verify model and inputs are on the same device and "
                            "run analyze_model() after changing layers."
                        ) from err2
            group.prune()
            try:
                tp.utils.remove_pruning_reparametrization(self.model)
            except Exception:  # pragma: no cover - safeguard against tp versions
                pass
        self.remove_hooks()
