from __future__ import annotations

from ultralytics.utils.torch_utils import get_flops, get_num_params

from pathlib import Path

from helper import (
    count_filters,
    file_size_mb,
    log_stats_comparison,
    count_params_in_layers,
    count_filters_in_layers,
    flops_in_layers,
)

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
        filters = count_filters(context.model.model)

        params_bb = count_params_in_layers(context.model, 0, 10)
        params_head = count_params_in_layers(context.model, 10, None)
        flops_bb = flops_in_layers(context.model, 0, 10)
        flops_head = flops_in_layers(context.model, 10, None)
        filters_bb = count_filters_in_layers(context.model, 0, 10)
        filters_head = count_filters_in_layers(context.model, 10, None)

        model_path = context.workdir / f"{self.dest}_model.pt"
        try:
            context.model.save(str(model_path))
        except Exception:  # pragma: no cover - best effort
            context.logger.exception("failed to save %s model", self.dest)
        size_mb = file_size_mb(model_path)
        stats = {
            "parameters": params,
            "flops": flops,
            "filters": filters,
            "model_size_mb": size_mb,
            "parameters_backbone": params_bb,
            "parameters_head": params_head,
            "flops_backbone": flops_bb,
            "flops_head": flops_head,
            "filters_backbone": filters_bb,
            "filters_head": filters_head,
        }
        if self.dest == "initial":
            context.initial_stats = stats
            context.metrics_mgr.record_pruning(
                {
                    "parameters": {"original": params},
                    "flops": {"original": flops},
                    "filters": {"original": filters},
                    "model_size_mb": {"original": size_mb},
                    "parameters_backbone": {"original": params_bb},
                    "parameters_head": {"original": params_head},
                    "flops_backbone": {"original": flops_bb},
                    "flops_head": {"original": flops_head},
                    "filters_backbone": {"original": filters_bb},
                    "filters_head": {"original": filters_head},
                }
            )
        else:
            context.pruned_stats = stats
            orig_params = context.initial_stats.get("parameters", params)
            orig_flops = context.initial_stats.get("flops", flops)
            orig_filters = context.initial_stats.get("filters", filters)
            orig_size = context.initial_stats.get("model_size_mb", size_mb)
            orig_p_bb = context.initial_stats.get("parameters_backbone", params_bb)
            orig_p_head = context.initial_stats.get("parameters_head", params_head)
            orig_f_bb = context.initial_stats.get("flops_backbone", flops_bb)
            orig_f_head = context.initial_stats.get("flops_head", flops_head)
            orig_fil_bb = context.initial_stats.get("filters_backbone", filters_bb)
            orig_fil_head = context.initial_stats.get("filters_head", filters_head)
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
                    "parameters_backbone": {
                        "pruned": params_bb,
                        "reduction": orig_p_bb - params_bb,
                        "reduction_percent": ((orig_p_bb - params_bb) / orig_p_bb * 100)
                        if orig_p_bb
                        else 0,
                    },
                    "parameters_head": {
                        "pruned": params_head,
                        "reduction": orig_p_head - params_head,
                        "reduction_percent": ((orig_p_head - params_head) / orig_p_head * 100)
                        if orig_p_head
                        else 0,
                    },
                    "flops_backbone": {
                        "pruned": flops_bb,
                        "reduction": orig_f_bb - flops_bb,
                        "reduction_percent": ((orig_f_bb - flops_bb) / orig_f_bb * 100)
                        if orig_f_bb
                        else 0,
                    },
                    "flops_head": {
                        "pruned": flops_head,
                        "reduction": orig_f_head - flops_head,
                        "reduction_percent": ((orig_f_head - flops_head) / orig_f_head * 100)
                        if orig_f_head
                        else 0,
                    },
                    "filters_backbone": {
                        "pruned": filters_bb,
                        "reduction": orig_fil_bb - filters_bb,
                        "reduction_percent": ((orig_fil_bb - filters_bb) / orig_fil_bb * 100)
                        if orig_fil_bb
                        else 0,
                    },
                    "filters_head": {
                        "pruned": filters_head,
                        "reduction": orig_fil_head - filters_head,
                        "reduction_percent": ((orig_fil_head - filters_head) / orig_fil_head * 100)
                        if orig_fil_head
                        else 0,
                    },
                }
            )
            ratio = orig_params / params if params else 0
            context.metrics_mgr.record_pruning({"compression_ratio": ratio})
            log_stats_comparison(context.initial_stats, context.pruned_stats, context.logger)

__all__ = ["CalcStatsStep"]
