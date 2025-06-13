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
        self.labels: List[torch.Tensor] = []
        self.layers: List[nn.Conv2d] = []

    # ------------------------------------------------------------------
    # Utility hooks
    # ------------------------------------------------------------------
    def _activation_hook(self, idx: int):
        def hook(_module: nn.Module, _input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            self.activations.setdefault(idx, []).append(output.detach())
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
        self.labels.append(y.detach())

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
        B, C, H, W = F.shape
        Ky = self._rbf_kernel(y.unsqueeze(1))
        scores = []
        for j in range(C):
            Kj = self._rbf_kernel(F[:, j, :, :])
            scores.append((Kj * Ky).mean())
        return torch.stack(scores)

    # ------------------------------------------------------------------
    # BasePruningMethod interface
    # ------------------------------------------------------------------
    def analyze_model(self) -> None:  # pragma: no cover - heavy dependency
        import torch_pruning as tp

        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, self.example_inputs)
        self.register_hooks()

    def generate_pruning_mask(self, ratio: float) -> None:
        if not self.activations or not self.labels:
            raise RuntimeError("No activations/labels collected. Run a forward pass first.")
        features = {idx: torch.cat(feats, dim=0) for idx, feats in self.activations.items()}
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
        num_prune = int(len(importance) * ratio)
        prune_order = torch.argsort(importance)[:num_prune]
        self.pruning_plan = {}
        for idx in prune_order.tolist():
            layer, ch = group_info[idx]
            self.pruning_plan.setdefault(layer, []).append(ch)

    def apply_pruning(self) -> None:  # pragma: no cover - heavy dependency
        if self.DG is None:
            raise RuntimeError("analyze_model must be called first")
        import torch_pruning as tp

        for layer, idxs in self.pruning_plan.items():
            self.DG.prune_layer(layer, idxs, dim=0)
        tp.utils.remove_pruning_reparametrization(self.model)
        self.remove_hooks()
