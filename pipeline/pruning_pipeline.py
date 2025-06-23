from __future__ import annotations

from typing import Any, Dict, Iterable, List
from pathlib import Path
import time

from .base_pipeline import BasePruningPipeline
from prune_methods.base import BasePruningMethod
from helper import (
    get_logger,
    Logger,
    MetricManager,
    count_filters,
    model_size_mb,
    log_stats_comparison,
    format_header,
    format_step,
)
from .model_reconfig import AdaptiveLayerReconfiguration
from .context import PipelineContext
from .step import PipelineStep

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


class PruningPipeline(BasePruningPipeline):
    """High level pipeline to orchestrate pruning of YOLO models."""

    def __init__(
        self,
        model_path: str,
        data: str,
        workdir: str = "runs/pruning",
        pruning_method: BasePruningMethod | None = None,
        logger: Logger | None = None,
        steps: Iterable[PipelineStep] | None = None,
    ) -> None:
        super().__init__(model_path, data, workdir, pruning_method, logger)
        self.model: YOLO | None = None
        self.metrics_mgr = MetricManager()
        self.metrics_csv: Path | None = None
        self.reconfigurator = AdaptiveLayerReconfiguration(logger=self.logger)
        self.steps: List[PipelineStep] = list(steps or [])

    # ------------------------------------------------------------------
    # Step-based execution
    # ------------------------------------------------------------------
    def run_pipeline(self) -> PipelineContext:
        """Execute all configured steps in order."""
        context = PipelineContext(
            model_path=self.model_path,
            data=self.data,
            workdir=self.workdir,
            pruning_method=self.pruning_method,
            logger=self.logger,
        )
        context.pipeline = self
        context.metrics_mgr = self.metrics_mgr
        total = len(self.steps)
        for idx, step in enumerate(self.steps, 1):
            title = format_step(idx, total, step.__class__.__name__)
            self.logger.info(format_header(title))
            start = time.time()
            step.run(context)
            elapsed = time.time() - start
            self.logger.info(format_header(f"{title} finished in {elapsed:.2f}s"))
        # sync results back to the pipeline instance
        self.model = context.model
        self.initial_stats = context.initial_stats
        self.pruned_stats = context.pruned_stats
        self.metrics = context.metrics
        self.metrics_mgr = context.metrics_mgr
        self.metrics_csv = self.metrics_mgr.to_csv(self.workdir / "metrics.csv")
        return context

    def load_model(self) -> None:
        """Load the YOLO model from ``self.model_path``."""
        self.logger.info("Loading model from %s", self.model_path)
        self.model = YOLO(self.model_path)

    def calc_initial_stats(self) -> Dict[str, float]:
        """Calculate parameter count and FLOPs before pruning."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Calculating initial statistics")
        params = get_num_params(self.model.model)
        flops = get_flops(self.model.model)
        filters = count_filters(self.model.model)
        size_mb = model_size_mb(self.model.model)
        self.initial_stats = {
            "parameters": params,
            "flops": flops,
            "filters": filters,
            "model_size_mb": size_mb,
        }
        self.metrics_mgr.record_pruning(
            {
                "parameters": {"original": params},
                "flops": {"original": flops},
                "filters": {"original": filters},
                "model_size_mb": {"original": size_mb},
            }
        )
        try:  # pragma: no cover - optional dependency
            from prune_methods.depgraph_hsic import DepgraphHSICMethod
        except Exception:
            DepgraphHSICMethod = None
        if (
            DepgraphHSICMethod is not None
            and isinstance(self.pruning_method, DepgraphHSICMethod)
        ):
            self.pruning_method.reset_records()
        return self.initial_stats


    def pretrain(self, *, device: str | int | list = 0, label_fn=None, **train_kwargs: Any) -> Dict[str, Any]:
        """Optional pretraining step to run before pruning."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Pretraining model")
        train_kwargs.setdefault("plots", True)
        if label_fn is None:
            label_fn = lambda batch: batch["cls"]
        self._register_label_callback(label_fn)
        original_model = self.model.model
        try:
            metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        finally:
            self._unregister_label_callback()
        model_changed = self.model.model is not original_model
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
            self.logger.debug("updated pruning method model reference")
            if model_changed:
                try:
                    if hasattr(self.pruning_method, "refresh_dependency_graph"):
                        # Model instance was replaced; rebuild dependencies
                        # without clearing hooks so recorded activations survive
                        self.pruning_method.refresh_dependency_graph()
                        self.logger.debug("refreshed pruning method dependency graph")
                    else:
                        self._sync_pruning_method(reanalyze=True)
                        self.logger.debug("reanalyzed pruning method model")
                except Exception:  # pragma: no cover - best effort
                    pass
        self.logger.debug(metrics)
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["pretrain"] = metrics
        return metrics or {}

    def analyze_structure(self) -> None:
        """Analyze model structure to guide pruning."""
        if self.pruning_method is not None:
            self.logger.info("Analyzing model structure")
            self.pruning_method.analyze_model()

    def generate_pruning_mask(self, ratio: float, dataloader: Any | None = None) -> None:
        """Generate pruning mask at the given ratio."""
        if self.pruning_method is not None:
            self.logger.info("Generating pruning mask at ratio %.2f", ratio)
            self.pruning_method.generate_pruning_mask(ratio, dataloader=dataloader)

    def apply_pruning(self, rebuild: bool = False) -> None:
        """Apply the generated pruning mask."""
        if self.pruning_method is not None:
            self.logger.info("Applying pruning")
            self.pruning_method.apply_pruning(rebuild=rebuild)

    def reconfigure_model(self, output_path: str | None = None) -> None:
        """Reconfigure the model after pruning."""
        self.logger.info("Reconfiguring model")
        if self.reconfigurator is not None:
            self.reconfigurator.reconfigure_model(self.model, output_path)
        else:
            self.logger.warning("No reconfigurator available")

    def calc_pruned_stats(self) -> Dict[str, float]:
        """Calculate parameter count and FLOPs after pruning."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Calculating pruned statistics")
        params = get_num_params(self.model.model)
        flops = get_flops(self.model.model)
        filters = count_filters(self.model.model)
        size_mb = model_size_mb(self.model.model)
        self.pruned_stats = {
            "parameters": params,
            "flops": flops,
            "filters": filters,
            "model_size_mb": size_mb,
        }
        self.metrics_mgr.record_pruning({
            "parameters": {"pruned": params},
            "flops": {"pruned": flops},
            "filters": {"pruned": filters},
            "model_size_mb": {"pruned": size_mb},
        })
        log_stats_comparison(self.initial_stats, self.pruned_stats, self.logger)
        return self.pruned_stats

    def finetune(self, *, device: str | int | list = 0, label_fn=None, **train_kwargs: Any) -> Dict[str, Any]:
        """Finetune the pruned model."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Finetuning pruned model")
        train_kwargs.setdefault("plots", True)
        if label_fn is None:
            label_fn = lambda batch: batch["cls"]
        self._register_label_callback(label_fn)
        try:
            metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        finally:
            self._unregister_label_callback()
        self.logger.debug(metrics)
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["finetune"] = metrics
        return metrics or {}

    def record_metrics(self) -> Dict[str, Any]:
        """Record all metrics."""
        return {
            "initial": self.initial_stats,
            "pruned": self.pruned_stats,
            "training": self.metrics,
        }

    def save_metrics_csv(self, path: str | Path) -> Path:
        """Save metrics to CSV file."""
        return self.metrics_mgr.to_csv(path)

    def save_model(self, path: str | Path) -> Path:
        """Save the pruned model."""
        if self.model is not None:
            self.model.save(path)
        return Path(path)

