from __future__ import annotations

from typing import Any

import torch


from .base import BasePruningMethod


class TorchRandomMethod(BasePruningMethod):
    """Prune using ``torch-pruning`` with random importance."""

    requires_reconfiguration = False

    def __init__(self, model: Any, workdir: str = "runs/pruning") -> None:
        super().__init__(model, workdir)
        self.example_inputs = torch.randn(1, 3, 640, 640)
        self.pruner: Any | None = None

    def analyze_model(self) -> None:
        """Build and store the dependency graph for ``self.model``."""
        self.logger.info("Analyzing model")
        import torch_pruning as tp
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(self.model, example_inputs=self._inputs_tuple())

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        import torch_pruning as tp
        importance = tp.importance.RandomImportance()
        pruner_cls = getattr(tp, "RandomPruner", getattr(tp, "MagnitudePruner", tp.pruner.algorithms.BasePruner))
        self.pruner = pruner_cls(
            self.model,
            example_inputs=self._inputs_tuple(),
            importance=importance,
            pruning_ratio=ratio,
            iterative_steps=1,
        )

    def apply_pruning(self, rebuild=False) -> None:
        self.logger.info("Applying pruning")
        if self.pruner is None:
            raise RuntimeError("generate_pruning_mask must be called first")
        self.pruner.step()

    def visualize_comparison(self) -> None:
        """Visualize baseline vs pruned metrics using base helper."""
        try:
            super().visualize_comparison()
        except Exception as exc:  # pragma: no cover - optional plotting
            self.logger.warning("Failed to visualize comparison: %s", exc)

    def visualize_pruned_filters(self) -> None:
        """Visualize pruned filters using base helper."""
        try:
            super().visualize_pruned_filters()
        except Exception as exc:  # pragma: no cover - optional plotting
            self.logger.warning("Failed to visualize pruned filters: %s", exc)

