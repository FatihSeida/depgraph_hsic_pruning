from __future__ import annotations

from typing import Any, Dict, Iterable, List
from pathlib import Path

from .base_pipeline import BasePruningPipeline
from prune_methods.base import BasePruningMethod
from helper import (
    get_logger,
    Logger,
    MetricManager,
    count_filters,
    model_size_mb,
    log_stats_comparison,
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
        self._label_callback = None

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
        context.metrics_mgr = self.metrics_mgr
        for step in self.steps:
            step.run(context)
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

    def _register_label_callback(self, label_fn) -> None:
        """Attach a callback to record labels for ``DepgraphHSICMethod``."""
        try:
            from prune_methods.depgraph_hsic import DepgraphHSICMethod  # local import to avoid heavy dependency at module import
        except Exception:  # pragma: no cover - dependency missing
            DepgraphHSICMethod = None

        if DepgraphHSICMethod is None or not isinstance(self.pruning_method, DepgraphHSICMethod):
            return

        if self._label_callback is None:
            def record_labels(trainer) -> None:  # pragma: no cover - heavy dependency
                batch = getattr(trainer, "batch", None)
                if isinstance(batch, dict) and "cls" in batch:
                    labels = label_fn(batch)
                    try:
                        import torch  # local import to avoid hard dependency at module import
                        if torch.is_tensor(labels) and len(labels) != batch["img"].shape[0]:
                            self.logger.warning(
                                "label_fn returned %d labels for batch size %d; labels are likely object-level and may cause activation/label mismatches",
                                len(labels),
                                batch["img"].shape[0],
                            )
                    except Exception:  # pragma: no cover - labels malformed or torch missing
                        pass
                    self.logger.debug(
                        "Adding labels for batch with shape %s", tuple(getattr(labels, "shape", []))
                    )
                    self.pruning_method.add_labels(labels)

            self._label_callback = record_labels

        try:
            existing = getattr(self.model, "callbacks", {}).get("on_train_batch_end", [])
            if self._label_callback not in existing:
                self.model.add_callback("on_train_batch_end", self._label_callback)
        except AttributeError:  # pragma: no cover - fallback for stubs
            pass

    def _unregister_label_callback(self) -> None:
        """Remove the label recording callback if present."""
        try:
            callbacks = getattr(self.model, "callbacks", {}).get("on_train_batch_end", [])
            if self._label_callback in callbacks:
                callbacks.remove(self._label_callback)
        except Exception:  # pragma: no cover - ignore errors
            pass
        self._label_callback = None

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
                        import torch  # local import to avoid heavy dependency at module import
                        try:
                            device = next(self.pruning_method.model.parameters()).device
                        except Exception:
                            device = torch.device("cpu") if hasattr(torch, "device") else "cpu"
                        example_inputs = getattr(self.pruning_method, "example_inputs", None)
                        if hasattr(torch, "is_tensor") and torch.is_tensor(example_inputs):
                            self.pruning_method.example_inputs = example_inputs.to(device)
                        self.pruning_method.analyze_model()
                        self.logger.debug("reanalyzed pruning method model")
                except Exception:  # pragma: no cover - best effort
                    pass
        self.logger.debug(metrics)
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["pretrain"] = metrics
        return metrics or {}

    def analyze_structure(self) -> None:
        """Analyze model structure to guide pruning."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Analyzing model structure")
        self.pruning_method.analyze_model()

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        """Generate pruning mask at ``ratio`` sparsity."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        if dataloader is None:
            try:
                trainer = getattr(self.model, "trainer", None)
                dataloader = getattr(trainer, "train_loader", None) or getattr(trainer, "train_dataloader", None) or getattr(trainer, "val_loader", None) or getattr(trainer, "val_dataloader", None)
            except Exception:  # pragma: no cover - best effort
                dataloader = None
        self.pruning_method.generate_pruning_mask(ratio, dataloader)

    def apply_pruning(self) -> None:
        """Apply the previously generated pruning mask to the model."""
        if self.pruning_method is None:
            raise NotImplementedError
        self.logger.info("Applying pruning mask")
        self.pruning_method.apply_pruning()

    def reconfigure_model(self, output_path: str | Path | None = None) -> None:
        """Reconfigure the model after pruning if necessary."""
        if self.pruning_method is None:
            raise NotImplementedError
        if not getattr(self.pruning_method, "requires_reconfiguration", True):
            return
        self.logger.info("Reconfiguring pruned model")
        if self.model is not None:
            self.model = self.reconfigurator.reconfigure_model(
                self.model, output_path=output_path
            )

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
        orig_params = self.initial_stats.get("parameters", params)
        orig_flops = self.initial_stats.get("flops", flops)
        orig_filters = self.initial_stats.get("filters", filters)
        orig_size = self.initial_stats.get("model_size_mb", size_mb)
        self.metrics_mgr.record_pruning(
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
                        # Preserve recorded activations while rebuilding dependencies
                        self.pruning_method.refresh_dependency_graph()
                        self.logger.debug("refreshed pruning method dependency graph")
                    else:
                        import torch  # local import to avoid heavy dependency at module import
                        try:
                            device = next(self.pruning_method.model.parameters()).device
                        except Exception:
                            device = torch.device("cpu") if hasattr(torch, "device") else "cpu"
                        example_inputs = getattr(self.pruning_method, "example_inputs", None)
                        if hasattr(torch, "is_tensor") and torch.is_tensor(example_inputs):
                            self.pruning_method.example_inputs = example_inputs.to(device)
                        self.pruning_method.analyze_model()
                        self.logger.debug("reanalyzed pruning method model")
                except Exception:  # pragma: no cover - best effort
                    pass
        self.logger.debug(metrics)
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["finetune"] = metrics
        return metrics or {}

    def record_metrics(self) -> Dict[str, Any]:
        """Return a dictionary containing all recorded metrics."""
        self.logger.info("Recording metrics")
        data = self.metrics_mgr.as_dict()
        data["initial"] = self.initial_stats
        data["pruned"] = self.pruned_stats
        return data

    def save_metrics_csv(self, path: str | Path) -> Path:
        """Persist recorded metrics to ``path`` using :class:`MetricManager`."""

        self.logger.info("Saving metrics CSV to %s", path)
        self.metrics_csv = self.metrics_mgr.to_csv(path)
        return self.metrics_csv

    def save_model(self, path: str | Path) -> Path:
        """Persist the current YOLO model to ``path``."""

        if self.model is None:
            raise ValueError("Model is not loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Saving model to %s", path)
        self.model.save(str(path))
        return path

