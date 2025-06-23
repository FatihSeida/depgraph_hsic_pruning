"""Utility functions for FLOPs calculation with graceful fallbacks."""
from __future__ import annotations

from typing import Any, Iterable

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from ultralytics.utils.torch_utils import (
        get_num_params as _get_num_params,
    )  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _get_num_params = None  # type: ignore


def _conv_hook(module: Any, _inp: Iterable[Any], output: Any, totals: list[int]) -> None:
    if torch is None or not hasattr(torch, "Tensor"):
        return
    Tensor = getattr(torch, "Tensor", None)
    if Tensor is None or not isinstance(output, Tensor):
        return
    out_h, out_w = output.shape[-2:]
    batch = output.shape[0]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
    totals[0] += int(batch * module.out_channels * out_h * out_w * kernel_ops)


def _linear_hook(module: Any, _inp: Iterable[Any], output: Any, totals: list[int]) -> None:
    if torch is None or not hasattr(torch, "Tensor"):
        return
    Tensor = getattr(torch, "Tensor", None)
    if Tensor is None or not isinstance(output, Tensor):
        return
    batch = output.shape[0] if output.dim() > 1 else 1
    totals[0] += int(batch * module.in_features * module.out_features)


def calculate_flops_manual(model: Any, imgsz: int | Iterable[int] = 640) -> float:
    """Return FLOPs in billions using a simple forward hook approach."""
    if torch is None or nn is None or not hasattr(torch, "zeros"):
        return 0.0
    if isinstance(imgsz, (int, float)):
        shape = (1, 3, int(imgsz), int(imgsz))
    else:
        s = list(imgsz)
        if len(s) == 2:
            shape = (1, 3, s[0], s[1])
        elif len(s) == 3:
            shape = (1, s[0], s[1], s[2])
        else:
            shape = tuple(s)
    try:
        device = next(model.parameters()).device
    except Exception:  # pragma: no cover - best effort
        device = None
    if device is not None:
        dummy = torch.zeros(*shape, device=device)
    else:
        dummy = torch.zeros(*shape)
    totals = [0]
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(lambda m, i, o, t=totals: _conv_hook(m, i, o, t)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(lambda m, i, o, t=totals: _linear_hook(m, i, o, t)))
    model.eval()
    try:
        with torch.no_grad():
            model(dummy)
    except Exception:
        # forward failed; return partial FLOPs collected so far
        for h in hooks:
            h.remove()
        return totals[0] * 2 / 1e9
    for h in hooks:
        h.remove()
    return totals[0] * 2 / 1e9


def get_flops_reliable(model: Any, imgsz: int | Iterable[int] = 640) -> float:
    """Return FLOPs using the built-in manual calculation."""
    return calculate_flops_manual(model, imgsz)


def get_num_params_reliable(model: Any) -> int:
    """Return parameter count using ``ultralytics`` if available, else manual count."""
    params = 0
    if _get_num_params is not None:
        try:
            params = int(_get_num_params(model))
        except Exception:  # pragma: no cover - best effort
            params = 0
    if not params:
        params = 0
        for p in getattr(model, "parameters", lambda: [])():
            try:
                params += p.numel()
            except Exception:  # pragma: no cover - non-tensor params
                continue
    return params


__all__ = [
    "calculate_flops_manual",
    "get_flops_reliable",
    "get_num_params_reliable",
]
