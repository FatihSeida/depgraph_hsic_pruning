"""Pruning based on L1-norm of convolutional filters."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .base import BasePruningMethod
from .utils import collect_backbone_convs


class L1NormMethod(BasePruningMethod):
    """Pruning method that removes filters with the smallest L1-norm."""

    def __init__(self, model: any, workdir: str = "runs/pruning") -> None:
        super().__init__(model, workdir)
        self.layers: List[Tuple[nn.Module, str, nn.BatchNorm2d | None]] = []
        self.ratio = 0.0

    def analyze_model(self) -> None:
        """Collect convolution layers from the first 10 backbone modules."""
        self.logger.info("Analyzing model")
        self.layers = collect_backbone_convs(self.model)

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        self.ratio = ratio
        self.masks = []
        for parent, attr, _ in self.layers:
            conv = getattr(parent, attr)
            scores = conv.weight.data.abs().sum(dim=(1, 2, 3))
            num_prune = int(conv.out_channels * ratio)
            num_prune = min(max(num_prune, 0), conv.out_channels - 1)
            self.logger.debug(
                "Layer %s pruning %d/%d channels",
                attr,
                num_prune,
                conv.out_channels,
            )
            _, idx = torch.sort(scores)
            mask = torch.ones(conv.out_channels, dtype=torch.bool)
            if num_prune > 0:
                mask[idx[:num_prune]] = False
            self.masks.append(mask)

    def apply_pruning(self, rebuild=False) -> None:
        self.logger.info("Applying pruning")
        for (parent, attr, bn), mask in zip(self.layers, self.masks):
            conv = getattr(parent, attr)
            keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
            if len(keep_idx) == conv.out_channels:
                self.logger.debug(
                    "Layer %s: %d -> %d channels (no pruning applied)",
                    attr,
                    conv.out_channels,
                    len(keep_idx),
                )
                continue
            self.logger.debug(
                "Layer %s: %d -> %d channels",
                attr,
                conv.out_channels,
                len(keep_idx),
            )
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

    def visualize_comparison(self) -> None:
        """Visualize baseline vs pruned metrics."""
        if not self.initial_stats or not self.pruned_stats:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            labels = ["baseline", "pruned"]
            params = [self.initial_stats.get("parameters", 0), self.pruned_stats.get("parameters", 0)]
            flops = [self.initial_stats.get("flops", 0), self.pruned_stats.get("flops", 0)]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].bar(labels, params)
            axes[0].set_title("Parameters")
            axes[1].bar(labels, flops)
            axes[1].set_title("FLOPs")
            plt.tight_layout()
            plt.savefig(self.workdir / "comparison.png")
            plt.close()
            
            self.logger.info("Comparison visualization saved to %s", self.workdir / "comparison.png")
        except Exception as e:
            self.logger.warning("Failed to create comparison visualization: %s", str(e))

    def visualize_pruned_filters(self) -> None:
        """Visualize which channels were pruned for each convolution layer.

        Produces a heatmap with layers on the y-axis and channel indices on the
        x-axis. Dark squares mark filters removed by pruning. The plot is saved
        as ``pruned_filters.png`` in ``self.workdir``.
        """

        if not self.masks or not self.layers:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            names = [name for _, name, _ in self.layers]
            pruned = [(~mask).cpu().numpy().astype(int) for mask in self.masks]
            max_channels = max(len(m) for m in pruned)

            matrix = np.zeros((len(pruned), max_channels), dtype=int)
            for i, m in enumerate(pruned):
                matrix[i, : len(m)] = m

            fig, ax = plt.subplots(figsize=(8, 0.5 * len(pruned) + 1))
            ax.imshow(matrix, cmap="Greys", aspect="auto")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("Channel index")
            ax.set_title("Pruned filter map (dark = pruned)")
            plt.tight_layout()
            plt.savefig(self.workdir / "pruned_filters.png")
            plt.close()

            self.logger.info(
                "Pruned filters visualization saved to %s",
                self.workdir / "pruned_filters.png",
            )
        except Exception as e:
            self.logger.warning(
                "Failed to create pruned filters visualization: %s", str(e)
            )
