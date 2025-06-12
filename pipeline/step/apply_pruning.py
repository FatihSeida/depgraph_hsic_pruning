from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class ApplyPruningStep(PipelineStep):
    """Apply the generated pruning mask to the model."""

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            raise NotImplementedError
        context.logger.info("Applying pruning mask")
        context.pruning_method.apply_pruning()

__all__ = ["ApplyPruningStep"]
