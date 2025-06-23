from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class CompareModelsStep(PipelineStep):
    """Visualize and save comparison plots using the pruning method."""

    def run(self, context: PipelineContext) -> None:
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        if context.pruning_method is None:
            return
        context.logger.info("Visualizing pruning results")
        
        # Call visualize_comparison if available
        if hasattr(context.pruning_method, 'visualize_comparison'):
            context.pruning_method.visualize_comparison()
        
        # Call visualize_pruned_filters if available
        if hasattr(context.pruning_method, 'visualize_pruned_filters'):
            context.pruning_method.visualize_pruned_filters()
            
        context.logger.info("Finished %s", step)

__all__ = ["CompareModelsStep"]
