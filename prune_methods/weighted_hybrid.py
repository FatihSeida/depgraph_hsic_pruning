from __future__ import annotations

"""Weighted Hybrid Criterion pruning."""

from typing import Any, List, Tuple

import torch
from torch import nn

from .base import BasePruningMethod
from .utils import collect_backbone_convs


class WeightedHybridMethod(BasePruningMethod):
    """Two-stage pruning: magnitude then redundancy reduction."""

    def __init__(self, model: Any, workdir: str = "runs/pruning", rate_norm: float = 0.5, rate_dist: float = 0.2) -> None:
        super().__init__(model, workdir)
        self.layers: List[Tuple[nn.Module, str, nn.BatchNorm2d | None]] = []
        self.rate_norm = rate_norm
        self.rate_dist = rate_dist
        self.ratio = 0.0

    def analyze_model(self) -> None:
        """Collect convolution layers from the first 10 backbone modules."""
        self.logger.info("Analyzing model")
        self.layers = collect_backbone_convs(self.model)

    def _pairwise_distance(self, W: torch.Tensor) -> torch.Tensor:
        norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.abs(torch.mm(norm, norm.t()))
        return 1 - sim

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        self.ratio = ratio
        self.masks = []
        for parent, attr, _ in self.layers:
            conv = getattr(parent, attr)
            scores = conv.weight.data.view(conv.out_channels, -1).norm(p=2, dim=1)
            num_keep1 = max(int(conv.out_channels * self.rate_norm), 1)
            _, keep1 = torch.sort(scores, descending=True)
            keep = keep1[:num_keep1]
            if len(keep) > 1:
                D = self._pairwise_distance(conv.weight.data[keep].view(len(keep), -1))
                remove_count = min(int(conv.out_channels * self.rate_dist), len(keep) - 1)
                keep_mask = torch.ones(len(keep), dtype=torch.bool)
                for _ in range(remove_count):
                    vals, _ = torch.min(D + torch.diag(torch.full((len(keep),), 10.0, device=D.device)), dim=1)
                    idx = torch.argmin(vals)
                    keep_mask[idx] = False
                    D[idx, :] = 10
                    D[:, idx] = 10
                keep = keep[keep_mask]
            num_keep = max(int(conv.out_channels * (1 - ratio)), 1)
            if len(keep) > num_keep:
                keep = keep[:num_keep]
            mask = torch.zeros(conv.out_channels, dtype=torch.bool)
            mask[keep] = True
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
