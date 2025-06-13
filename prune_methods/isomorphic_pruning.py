from __future__ import annotations

"""Isomorphic pruning method based on dependency graphs."""

from typing import Any

import torch
from torch import nn

from .base import BasePruningMethod


class IsomorphicPruningMethod(BasePruningMethod):
    """Pruning that preserves layer shapes via ``torch-pruning``."""

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning", round_to: int | None = None) -> None:
        super().__init__(model, workdir)
        self.example_inputs = torch.randn(1, 3, 640, 640)
        self.round_to = round_to
        self.pruner = None

    def analyze_model(self) -> None:  # pragma: no cover - heavy dependency
        import torch_pruning as tp
        tp.DependencyGraph().build_dependency(self.model, self.example_inputs)

    def generate_pruning_mask(self, ratio: float) -> None:  # pragma: no cover - heavy dependency
        import torch_pruning as tp
        importance = tp.importance.MagnitudeImportance(p=2)
        self.pruner = tp.MagnitudePruner(
            self.model,
            example_inputs=self.example_inputs,
            importance=importance,
            global_pruning=True,
            ch_sparsity=ratio,
            round_to=self.round_to,
        )

    def apply_pruning(self) -> None:  # pragma: no cover - heavy dependency
        if self.pruner is None:
            raise RuntimeError("generate_pruning_mask must be called first")
        self.pruner.step()
        for m in self.model.modules():
            reparam = getattr(m, "reparam", None)
            if reparam is not None:
                try:
                    reparam.prune_finalize()
                except Exception:
                    pass
