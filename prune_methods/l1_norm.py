"""Pruning based on L1-norm of convolutional filters."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .base import BasePruningMethod


class L1NormPruningMethod(BasePruningMethod):
    """Pruning method that removes filters with the smallest L1-norm."""

    def __init__(self, model: any, workdir: str = "runs/pruning") -> None:
        super().__init__(model, workdir)
        self.layers: List[Tuple[nn.Module, str, nn.BatchNorm2d | None]] = []
        self.ratio = 0.0

    def analyze_model(self) -> None:
        """Collect convolution layers from the first 10 backbone modules."""
        backbone = list(self.model.model[:10])
        for module in backbone:
            for name, m in module.named_modules():
                if isinstance(m, nn.Conv2d):
                    parent = module.get_submodule(".".join(name.split(".")[:-1])) if "." in name else module
                    bn = getattr(parent, "bn", None)
                    if not isinstance(bn, nn.BatchNorm2d):
                        bn = None
                    self.layers.append((parent, name.split(".")[-1], bn))

    def generate_pruning_mask(self, ratio: float) -> None:
        self.ratio = ratio
        self.masks = []
        for parent, attr, _ in self.layers:
            conv = getattr(parent, attr)
            scores = conv.weight.data.abs().sum(dim=(1, 2, 3))
            num_prune = int(conv.out_channels * ratio)
            num_prune = min(max(num_prune, 0), conv.out_channels - 1)
            _, idx = torch.sort(scores)
            mask = torch.ones(conv.out_channels, dtype=torch.bool)
            if num_prune > 0:
                mask[idx[:num_prune]] = False
            self.masks.append(mask)

    def apply_pruning(self) -> None:
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
