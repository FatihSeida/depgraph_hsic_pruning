from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class AnalyzeModelStep(PipelineStep):
    """Analyze the model using the configured pruning method."""

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            raise NotImplementedError
        context.logger.info("Analyzing model structure")
        context.pruning_method.analyze_model()


class AnalyzeAfterTrainingStep(PipelineStep):
    """Rebuild the dependency graph after a training phase."""

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            raise NotImplementedError
        context.logger.info("Reanalyzing model after training")
        context.pruning_method.analyze_model()


__all__ = ["AnalyzeModelStep", "AnalyzeAfterTrainingStep"]
