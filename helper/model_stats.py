"""Utility functions for computing model statistics."""
from __future__ import annotations

from typing import Any

import os
from pathlib import Path

import logging

from .flops_utils import get_flops_reliable

try:
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover - torch may not be installed
    nn = None  # type: ignore


def count_filters(model: Any) -> int:
    """Return the total number of convolution filters in ``model``."""
    total = 0
    modules = getattr(model, "modules", lambda: [])()
    for m in modules:
        if nn is not None and isinstance(m, nn.Conv2d):
            total += int(getattr(m, "out_channels", 0))
        else:
            # Fallback: check for ``out_channels`` attribute
            if hasattr(m, "out_channels"):
                total += int(getattr(m, "out_channels", 0))
    return total


def model_size_mb(model: Any) -> float:
    """Approximate model size in megabytes based on parameters."""
    size_bytes = 0
    for p in getattr(model, "parameters", lambda: [])():
        try:
            size_bytes += p.numel() * p.element_size()
        except Exception as exc:  # pragma: no cover - for non-tensor params
            logging.debug("model_size_mb parameter error: %s", exc)
            continue
    return size_bytes / (1024 * 1024)


def count_params_in_layers(model: Any, start: int, end: int | None = None) -> int:
    """Return parameter count for ``model.model[start:end]``."""
    total = 0
    modules = list(getattr(model, "model", [])[start:end])
    for m in modules:
        for p in getattr(m, "parameters", lambda: [])():
            try:
                total += p.numel()
            except Exception as exc:  # pragma: no cover - non-tensor params
                logging.debug("count_params_in_layers error: %s", exc)
                continue
    return total


def count_filters_in_layers(model: Any, start: int, end: int | None = None) -> int:
    """Return convolution filter count for ``model.model[start:end]``."""
    total = 0
    modules = list(getattr(model, "model", [])[start:end])
    for module in modules:
        for sub in getattr(module, "modules", lambda: [])():
            if nn is not None and hasattr(nn, "Conv2d") and isinstance(sub, nn.Conv2d):
                total += int(getattr(sub, "out_channels", 0))
            elif hasattr(sub, "out_channels"):
                total += int(getattr(sub, "out_channels", 0))
    return total


def flops_in_layers(model: Any, start: int, end: int | None = None) -> float:
    """Return FLOPs for ``model.model[start:end]`` using ``get_flops_reliable``."""
    modules = list(getattr(model, "model", [])[start:end])
    if nn is not None and hasattr(nn, "Sequential"):
        container = nn.Sequential(*modules)
    else:
        from types import SimpleNamespace
        container = SimpleNamespace(model=modules)
    return float(get_flops_reliable(container))



def log_stats_comparison(initial: dict, pruned: dict, logger: Any) -> None:
    """Log a table comparing ``initial`` and ``pruned`` model statistics.

    Parameters
    ----------
    initial : dict
        Statistics before pruning.
    pruned : dict
        Statistics after pruning.
    logger : Any
        Logger instance providing an ``info`` method.
    """

    headers = ["metric", "initial", "pruned", "reduction", "%"]
    rows = [headers]
    metrics = ["parameters", "flops", "filters", "model_size_mb"]

    def fmt(val: float | int) -> str:
        if isinstance(val, float) and not val.is_integer():
            return f"{val:.2f}"
        return f"{int(val):,}"

    for key in metrics:
        orig = float(initial.get(key, 0))
        new = float(pruned.get(key, 0))
        red = orig - new
        pct = (red / orig * 100) if orig else 0.0
        rows.append([key, fmt(orig), fmt(new), fmt(red), f"{pct:.1f}%"])

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    lines = [
        " " .join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
        for row in rows
    ]
    table = "\n".join(lines)
    logger.info("\n%s", table)


def file_size_mb(path: str | Path) -> float:
    """Return size of ``path`` in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


__all__ = [
    "count_filters",
    "model_size_mb",
    "count_params_in_layers",
    "count_filters_in_layers",
    "flops_in_layers",
    "file_size_mb",
    "log_stats_comparison",
]
