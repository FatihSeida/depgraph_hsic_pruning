from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class CompareModelsStep(PipelineStep):
    """Visualize and save comparison plots using the pruning method."""

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            return
        context.logger.info("Visualizing pruning results")
        context.pruning_method.visualize_comparison()
        context.pruning_method.visualize_pruned_filters()

__all__ = ["CompareModelsStep"]
