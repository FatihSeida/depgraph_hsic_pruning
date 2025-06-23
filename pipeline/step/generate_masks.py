from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep
from typing import Any


class GenerateMasksStep(PipelineStep):
    """Generate pruning masks at a given sparsity ratio."""

    def __init__(self, ratio: float, dataloader: Any | None = None) -> None:
        self.ratio = ratio
        self.dataloader = dataloader

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            raise NotImplementedError
        dataloader = self.dataloader or getattr(context, "dataloader", None)
        context.pruning_method.generate_pruning_mask(self.ratio, dataloader=dataloader)

__all__ = ["GenerateMasksStep"]
