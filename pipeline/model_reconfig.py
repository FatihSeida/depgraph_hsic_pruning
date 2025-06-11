from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from helper import get_logger, Logger
from ultralytics_pruning import YOLO


class ModelReconfiguration:
    """Base interface for model reconfiguration strategies."""

    def reconfigure_model(
        self, pruned_model: YOLO, output_path: Optional[str] = None, mismatch_callback=None
    ) -> YOLO:
        raise NotImplementedError


class AdaptiveLayerReconfiguration(ModelReconfiguration):
    """Reconfigure pruned models by adapting convolution channels."""

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger = logger or get_logger()

    def reconfigure_model(
        self, pruned_model: YOLO, output_path: Optional[str] = None, mismatch_callback=None
    ) -> YOLO:
        self.logger.info("Applying Adaptive Layer Reconfiguration")
        model = pruned_model.model
        backbone = list(model.model[:10])

        # Gather output channels for backbone modules
        channel_map = {}
        for i, module in enumerate(backbone):
            last_conv = None
            name = None
            for n, m in module.named_modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
                    name = n
            if last_conv is not None:
                channel_map[i] = last_conv.out_channels
                self.logger.debug(f"Layer {i} output channels: {last_conv.out_channels}")

        # Adjust mismatched channels within backbone
        for i in range(len(backbone) - 1):
            out_ch = channel_map.get(i)
            if out_ch is None:
                continue
            next_module = backbone[i + 1]
            for name, sub in next_module.named_modules():
                if isinstance(sub, nn.Conv2d):
                    in_ch = sub.in_channels
                    if in_ch != out_ch:
                        if mismatch_callback:
                            mismatch_callback(i + 1, in_ch, out_ch)
                        self.logger.debug(
                            f"Adapting module {i+1} conv {name}: {in_ch} -> {out_ch}"
                        )
                        new_conv = nn.Conv2d(
                            out_ch,
                            sub.out_channels,
                            kernel_size=sub.kernel_size,
                            stride=sub.stride,
                            padding=sub.padding,
                            dilation=sub.dilation,
                            bias=sub.bias is not None,
                            padding_mode=sub.padding_mode,
                            groups=1,
                        )
                        nn.init.kaiming_normal_(new_conv.weight)
                        if sub.bias is not None:
                            nn.init.zeros_(new_conv.bias)
                        parent = next_module
                        attr = name
                        if "." in name:
                            *parents, attr = name.split(".")
                            parent = next_module.get_submodule(".".join(parents))
                        setattr(parent, attr, new_conv)
                    break

        # Align head with backbone output
        if len(model.model) > 10 and 9 in channel_map:
            head = model.model[10]
            out_ch = channel_map[9]
            for name, sub in head.named_modules():
                if isinstance(sub, nn.Conv2d):
                    in_ch = sub.in_channels
                    if in_ch != out_ch:
                        if mismatch_callback:
                            mismatch_callback("head", in_ch, out_ch)
                        self.logger.debug(
                            f"Adapting head conv {name}: {in_ch} -> {out_ch}"
                        )
                        new_conv = nn.Conv2d(
                            out_ch,
                            sub.out_channels,
                            kernel_size=sub.kernel_size,
                            stride=sub.stride,
                            padding=sub.padding,
                            dilation=sub.dilation,
                            bias=sub.bias is not None,
                            padding_mode=sub.padding_mode,
                            groups=1,
                        )
                        nn.init.kaiming_normal_(new_conv.weight)
                        if sub.bias is not None:
                            nn.init.zeros_(new_conv.bias)
                        parent = head
                        attr = name
                        if "." in name:
                            *parents, attr = name.split(".")
                            parent = head.get_submodule(".".join(parents))
                        setattr(parent, attr, new_conv)
                    break

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pruned_model.save(output_path)

        return pruned_model

