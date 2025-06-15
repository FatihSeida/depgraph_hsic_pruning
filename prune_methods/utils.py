"""Utility helpers for pruning methods."""

from __future__ import annotations

from typing import Any, List, Tuple

from torch import nn


def collect_backbone_convs(model: Any, num_modules: int = 10) -> List[Tuple[nn.Module, str, nn.BatchNorm2d | None]]:
    """Return ``(parent, attr, bn)`` tuples for conv layers in the backbone.

    Parameters
    ----------
    model : Any
        Model with a ``model`` attribute containing modules.
    num_modules : int, optional
        Number of modules from ``model.model`` to inspect, by default ``10``.
    """
    layers: List[Tuple[nn.Module, str, nn.BatchNorm2d | None]] = []
    backbone = list(model.model[:num_modules])
    for module in backbone:
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                parent = module.get_submodule(".".join(name.split(".")[:-1])) if "." in name else module
                bn = getattr(parent, "bn", None)
                if not isinstance(bn, nn.BatchNorm2d):
                    bn = None
                layers.append((parent, name.split(".")[-1], bn))
    return layers
