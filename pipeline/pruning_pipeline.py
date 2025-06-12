from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .base_pipeline import BasePruningPipeline
from prune_methods.base import BasePruningMethod
from helper import get_logger, Logger
from .model_reconfig import AdaptiveLayerReconfiguration
from .context import PipelineContext
from .step import PipelineStep

from ultralytics_pruning import YOLO
from ultralytics_pruning.utils.torch_utils import get_flops, get_num_params


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
        for step in self.steps:
            step.run(context)
        # sync results back to the pipeline instance
        self.model = context.model
        self.initial_stats = context.initial_stats
        self.pruned_stats = context.pruned_stats
        self.metrics = context.metrics
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
        self.initial_stats = {"parameters": params, "flops": flops}
        return self.initial_stats

    def pretrain(self, *, device: str | int | list = 0, **train_kwargs: Any) -> Dict[str, Any]:
        """Optional pretraining step to run before pruning."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Pretraining model")
        metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        self.metrics["pretrain"] = metrics
        return metrics or {}

    def analyze_structure(self) -> None:
        """Analyze model structure to guide pruning."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Analyzing model structure")
        self.pruning_method.analyze_model()

    def generate_pruning_mask(self, ratio: float) -> None:
        """Generate pruning mask at ``ratio`` sparsity."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        self.pruning_method.generate_pruning_mask(ratio)

    def apply_pruning(self) -> None:
        """Apply the previously generated pruning mask to the model."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Applying pruning mask")
        self.pruning_method.apply_pruning()

    def reconfigure_model(self) -> None:
        """Reconfigure the model after pruning if necessary."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Reconfiguring pruned model")
        if self.model is not None:
            self.reconfigurator.reconfigure_model(self.model)

    def calc_pruned_stats(self) -> Dict[str, float]:
        """Calculate parameter count and FLOPs after pruning."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Calculating pruned statistics")
        params = get_num_params(self.model.model)
        flops = get_flops(self.model.model)
        self.pruned_stats = {"parameters": params, "flops": flops}
        return self.pruned_stats

    def finetune(self, *, device: str | int | list = 0, **train_kwargs: Any) -> Dict[str, Any]:
        """Finetune the pruned model."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Finetuning pruned model")
        metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        self.metrics["finetune"] = metrics
        return metrics or {}

    def record_metrics(self) -> Dict[str, Any]:
        """Return a dictionary containing training and pruning statistics."""
        self.logger.info("Recording metrics")
        return {
            "initial": self.initial_stats,
            "pruned": self.pruned_stats,
            "training": self.metrics,
        }

