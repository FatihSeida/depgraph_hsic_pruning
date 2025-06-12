from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class GenerateMasksStep(PipelineStep):
    """Generate pruning masks at a given sparsity ratio."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            raise NotImplementedError
        context.logger.info("Generating pruning mask at ratio %.2f", self.ratio)
        context.pruning_method.generate_pruning_mask(self.ratio)

__all__ = ["GenerateMasksStep"]
