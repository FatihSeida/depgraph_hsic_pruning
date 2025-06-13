from __future__ import annotations

from typing import Any

import torch

from .base import BasePruningMethod


class DepgraphMethod(BasePruningMethod):
    """Pruning using ``torch-pruning`` with a dependency graph."""

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning") -> None:
        super().__init__(model, workdir)
        self.example_inputs = torch.randn(1, 3, 640, 640)
        self.pruner: Any | None = None

    def analyze_model(self) -> None:
        """Build and store the dependency graph for ``self.model``."""
        import torch_pruning as tp  # local import
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, self.example_inputs)

    def generate_pruning_mask(self, ratio: float) -> None:
        import torch_pruning as tp
        importance = tp.importance.MagnitudeImportance(p=2)
        pruner_cls = getattr(tp, "MagnitudePruner", tp.pruner.algorithms.BasePruner)
        self.pruner = pruner_cls(
            self.model,
            example_inputs=self.example_inputs,
            importance=importance,
            pruning_ratio=ratio,
            iterative_steps=1,
        )

    def apply_pruning(self) -> None:
        if self.pruner is None:
            raise RuntimeError("generate_pruning_mask must be called first")
        self.pruner.step()

