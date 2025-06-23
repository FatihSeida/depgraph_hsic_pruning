from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class CompareModelsStep(PipelineStep):
    """Visualize and save comparison plots using the pruning method."""

    def run(self, context: PipelineContext) -> None:
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        pipeline = getattr(context, "pipeline", None)
        if pipeline is not None:
            pipeline.visualize_results()
        else:
            context.logger.debug("No pipeline attached; skipping visualization")
        context.logger.info("Finished %s", step)

__all__ = ["CompareModelsStep"]
