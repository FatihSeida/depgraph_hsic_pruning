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
            context.metrics_mgr.record_pruning({
                "parameters": {"original": params},
                "flops": {"original": flops},
            })
        else:
            context.pruned_stats = stats
            orig_params = context.initial_stats.get("parameters", params)
            orig_flops = context.initial_stats.get("flops", flops)
            context.metrics_mgr.record_pruning({
                "parameters": {
                    "pruned": params,
                    "reduction": orig_params - params,
                    "reduction_percent": ((orig_params - params) / orig_params * 100) if orig_params else 0,
                },
                "flops": {
                    "pruned": flops,
                    "reduction": orig_flops - flops,
                    "reduction_percent": ((orig_flops - flops) / orig_flops * 100) if orig_flops else 0,
                },
            })

__all__ = ["CalcStatsStep"]
