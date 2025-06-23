from __future__ import annotations

"""HSIC-Lasso channel pruning implementation."""

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from sklearn.linear_model import LassoLars


def _rbf_kernel(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute a centred RBF kernel matrix."""
    b = x.shape[0]
    flat = x.view(b, -1)
    dist = torch.cdist(flat, flat)
    k = torch.exp(-gamma * dist.pow(2))
    h = torch.eye(b, device=k.device) - 1.0 / b
    return h @ k @ h


def compute_channel_wise_hsic(
    features: torch.Tensor, labels: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Return HSIC values between each channel in ``features`` and ``labels``."""

    b, c, _, _ = features.shape
    ky = _rbf_kernel(labels.unsqueeze(1), gamma)
    scores = []
    for j in range(c):
        kj = _rbf_kernel(features[:, j, :, :], gamma)
        scores.append((kj * ky).mean())
    return torch.stack(scores)


def solve_hsic_lasso(
    features: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    alpha: float = 0.001,
) -> torch.Tensor:
    """Compute HSIC-Lasso importance for each channel."""

    hsic = compute_channel_wise_hsic(features, labels, gamma)
    x = features.mean(dim=(2, 3)).cpu().numpy()
    y = labels.view(len(labels), -1).mean(dim=1).cpu().numpy()
    lasso = LassoLars(alpha=alpha)
    lasso.fit(x, y)
    coef = torch.tensor(lasso.coef_)
    return coef.abs() * hsic.cpu()

from .base import BasePruningMethod


class HSICLassoMethod(BasePruningMethod):
    """Prune channels using HSIC criterion with L1 sparsity."""

    def __init__(self, model: Any, workdir: str = "runs/pruning", gamma: float = 1.0) -> None:
        super().__init__(model, workdir)
        self.layers: List[Tuple[nn.Module, str, nn.BatchNorm2d | None]] = []
        self.layer_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.gamma = gamma
        self.ratio = 0.0

    def analyze_model(self) -> None:
        """Collect convolution layers from the first 10 backbone modules."""
        self.logger.info("Analyzing model")
        backbone = list(self.model.model[:10])
        for module in backbone:
            for name, m in module.named_modules():
                if isinstance(m, nn.Conv2d):
                    parent = module.get_submodule(".".join(name.split(".")[:-1])) if "." in name else module
                    bn = getattr(parent, "bn", None)
                    if not isinstance(bn, nn.BatchNorm2d):
                        bn = None
                    self.layers.append((parent, name.split(".")[-1], bn))

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------
    def set_layer_data(self, idx: int, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Attach feature maps and labels for layer ``idx``."""
        self.layer_data[idx] = (features, labels)

    # ------------------------------------------------------------------
    # Pruning logic
    # ------------------------------------------------------------------

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        self.ratio = ratio
        self.masks = []
        for idx, (parent, attr, _) in enumerate(self.layers):
            conv = getattr(parent, attr)
            if idx in self.layer_data:
                F, y = self.layer_data[idx]
                scores = solve_hsic_lasso(F, y, gamma=self.gamma)
            else:
                scores = conv.weight.data.abs().sum(dim=(1, 2, 3))
            num_keep = max(int(conv.out_channels * (1 - ratio)), 1)
            _, order = torch.sort(scores, descending=True)
            mask = torch.zeros(conv.out_channels, dtype=torch.bool)
            mask[order[:num_keep]] = True
            self.masks.append(mask)

    def apply_pruning(self, rebuild=False) -> None:
        self.logger.info("Applying pruning")
        for (parent, attr, bn), mask in zip(self.layers, self.masks):
            conv = getattr(parent, attr)
            keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
            if len(keep_idx) == conv.out_channels:
                continue
            new_conv = nn.Conv2d(
                conv.in_channels,
                len(keep_idx),
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=conv.bias is not None,
                padding_mode=conv.padding_mode,
            )
            new_conv.weight.data = conv.weight.data[keep_idx].clone()
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[keep_idx].clone()
            setattr(parent, attr, new_conv)
            if bn is not None:
                new_bn = nn.BatchNorm2d(len(keep_idx))
                new_bn.weight.data = bn.weight.data[keep_idx].clone()
                new_bn.bias.data = bn.bias.data[keep_idx].clone()
                new_bn.running_mean = bn.running_mean[keep_idx].clone()
                new_bn.running_var = bn.running_var[keep_idx].clone()
                setattr(parent, "bn", new_bn)
