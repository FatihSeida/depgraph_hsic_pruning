"""Utility functions for computing model statistics."""
from __future__ import annotations

from typing import Any

import logging

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


__all__ = ["count_filters", "model_size_mb", "log_stats_comparison"]
