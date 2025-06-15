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
    """Prune channel groups via HSIC and ``torch-pruning`` dependency graph."""

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning", gamma: float = 1.0) -> None:
        super().__init__(model, workdir)
        self.gamma = gamma
        self.example_inputs = torch.randn(1, 3, 640, 640)
        self.DG = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.layer_shapes: Dict[int, Tuple[int, int]] = {}
        self.num_activations: Dict[int, int] = {}
        self.labels: List[torch.Tensor] = []
        self.layers: List[nn.Conv2d] = []
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
        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on convolution layers."""
        self.handles = []
        self.layers = []
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.layers.append(m)
                self.handles.append(m.register_forward_hook(self._activation_hook(idx)))
                idx += 1

    def remove_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def add_labels(self, y: torch.Tensor) -> None:
        """Store labels observed during a forward pass."""
        processed = y.detach().cpu()
        self.labels.append(processed)
        self.logger.debug("Recorded label with shape %s", tuple(processed.shape))

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

    def _hsic_scores(self, F: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if F.shape[0] != y.shape[0]:
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

        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, self.example_inputs)
        self.register_hooks()
        self._build_adjacency()
        self._build_channel_groups()

    def generate_pruning_mask(self, ratio: float) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        if not self.activations or not self.labels:
            raise RuntimeError("No activations/labels collected. Run a forward pass first.")
        for idx, count in self.num_activations.items():
            self.logger.debug("Layer %d recorded %d activations", idx, count)
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
            scores = self._hsic_scores(F, y)
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
        self.pruning_plan = {}
        for score, chans in group_scores:
            if removed >= target:
                break
            for layer, ch in chans:
                self.pruning_plan.setdefault(layer, []).append(ch)
            removed += len(chans)

    def apply_pruning(self) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Applying pruning")
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")
        import torch_pruning as tp

        for layer, idxs in self.pruning_plan.items():
            unique = sorted(set(idxs))
            self.DG.prune_layer(layer, unique, dim=0)
        tp.utils.remove_pruning_reparametrization(self.model)
        self.remove_hooks()
