from __future__ import annotations

"""Isomorphic pruning using dependency graphs.

Filters are removed per layer based on the L1 norm of their weights while
keeping tensor shapes valid through ``torch-pruning``'s dependency graph.
"""

from typing import Any

import torch

from .base import BasePruningMethod


class IsomorphicMethod(BasePruningMethod):
    """Pruning that preserves layer shapes via ``torch-pruning``."""

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning", round_to: int | None = None) -> None:
        super().__init__(model, workdir)
        self.example_inputs = torch.randn(1, 3, 640, 640)
        self.round_to = round_to
        self.pruner = None

    def analyze_model(self) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Analyzing model")
        import torch_pruning as tp
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        import torch_pruning as tp
        importance = tp.importance.MagnitudeImportance(p=1)
        self.pruner = tp.MagnitudePruner(
            self.model,
            example_inputs=self._inputs_tuple(),
            importance=importance,
            global_pruning=False,
            pruning_ratio=ratio,
            round_to=self.round_to,
        )

    def apply_pruning(self, rebuild=False) -> None:  # pragma: no cover - heavy dependency
        self.logger.info("Applying pruning")
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
