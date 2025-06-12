from __future__ import annotations

from ultralytics.utils.torch_utils import get_flops, get_num_params

from ..context import PipelineContext
from . import PipelineStep


class CalcStatsStep(PipelineStep):
    """Calculate FLOPs and parameter count for the current model."""

    def __init__(self, dest: str) -> None:
        self.dest = dest

    def run(self, context: PipelineContext) -> None:
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Calculating %s statistics", self.dest)
        params = get_num_params(context.model.model)
        flops = get_flops(context.model.model)
        stats = {"parameters": params, "flops": flops}
        if self.dest == "initial":
            context.initial_stats = stats
        else:
            context.pruned_stats = stats

__all__ = ["CalcStatsStep"]
