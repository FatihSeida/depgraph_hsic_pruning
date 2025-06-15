from __future__ import annotations

from ultralytics.utils.torch_utils import get_flops, get_num_params

from helper import count_filters, model_size_mb, log_stats_comparison

from ..context import PipelineContext
from . import PipelineStep


class CalcStatsStep(PipelineStep):
    """Calculate FLOPs and parameter count for the current model."""

    def __init__(self, dest: str) -> None:
        self.dest = dest

    def run(self, context: PipelineContext) -> None:
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Calculating %s statistics", self.dest)
        params = get_num_params(context.model.model)
        flops = get_flops(context.model.model)
        filters = count_filters(context.model.model)
        size_mb = model_size_mb(context.model.model)
        stats = {
            "parameters": params,
            "flops": flops,
            "filters": filters,
            "model_size_mb": size_mb,
        }
        if self.dest == "initial":
            context.initial_stats = stats
            context.metrics_mgr.record_pruning(
                {
                    "parameters": {"original": params},
                    "flops": {"original": flops},
                    "filters": {"original": filters},
                    "model_size_mb": {"original": size_mb},
                }
            )
        else:
            context.pruned_stats = stats
            orig_params = context.initial_stats.get("parameters", params)
            orig_flops = context.initial_stats.get("flops", flops)
            orig_filters = context.initial_stats.get("filters", filters)
            orig_size = context.initial_stats.get("model_size_mb", size_mb)
            context.metrics_mgr.record_pruning(
                {
                    "parameters": {
                        "pruned": params,
                        "reduction": orig_params - params,
                        "reduction_percent": ((orig_params - params) / orig_params * 100)
                        if orig_params
                        else 0,
                    },
                    "flops": {
                        "pruned": flops,
                        "reduction": orig_flops - flops,
                        "reduction_percent": ((orig_flops - flops) / orig_flops * 100)
                        if orig_flops
                        else 0,
                    },
                    "filters": {
                        "pruned": filters,
                        "reduction": orig_filters - filters,
                        "reduction_percent": ((orig_filters - filters) / orig_filters * 100)
                        if orig_filters
                        else 0,
                    },
                    "model_size_mb": {
                        "pruned": size_mb,
                        "reduction": orig_size - size_mb,
                        "reduction_percent": ((orig_size - size_mb) / orig_size * 100)
                        if orig_size
                        else 0,
                    },
                }
            )
            log_stats_comparison(context.initial_stats, context.pruned_stats, context.logger)
        context.logger.info("Finished %s", step)

__all__ = ["CalcStatsStep"]
