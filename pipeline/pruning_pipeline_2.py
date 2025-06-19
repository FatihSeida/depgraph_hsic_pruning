from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from .base_pipeline import BasePruningPipeline
from prune_methods.depgraph_hsic import DepgraphHSICMethod
from helper import (
    Logger,
    MetricManager,
    count_filters,
    model_size_mb,
    log_stats_comparison,
)


class PruningPipeline2(BasePruningPipeline):
    """Simple pipeline specialised for :class:`DepgraphHSICMethod`."""

    def __init__(
        self,
        model_path: str,
        data: str,
        workdir: str = "runs/pruning",
        pruning_method: DepgraphHSICMethod | None = None,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(model_path, data, workdir, pruning_method, logger)
        self.model: YOLO | None = None
        self.metrics_mgr = MetricManager()
        self.metrics_csv: Path | None = None
        self._label_callback = None

    # ------------------------------------------------------------------
    # Helper callbacks
    # ------------------------------------------------------------------
    def _register_label_callback(self, label_fn) -> None:
        """Attach a callback to record labels for ``DepgraphHSICMethod``."""
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            return

        if self._label_callback is None:
            def record_labels(trainer) -> None:  # pragma: no cover - heavy deps
                batch = getattr(trainer, "batch", None)
                if isinstance(batch, dict) and "cls" in batch:
                    labels = label_fn(batch)
                    try:
                        import torch  # local import
                        if torch.is_tensor(labels) and len(labels) != batch["img"].shape[0]:
                            self.logger.warning(
                                "label_fn returned %d labels for batch size %d; labels may be mismatched",
                                len(labels),
                                batch["img"].shape[0],
                            )
                    except Exception:
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
        except AttributeError:
            pass

    def _unregister_label_callback(self) -> None:
        try:
            callbacks = getattr(self.model, "callbacks", {}).get("on_train_batch_end", [])
            if self._label_callback in callbacks:
                callbacks.remove(self._label_callback)
        except Exception:
            pass
        self._label_callback = None

    # ------------------------------------------------------------------
    # BasePruningPipeline interface
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        self.logger.info("Loading model from %s", self.model_path)
        self.model = YOLO(self.model_path)
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
        self.logger.info("Model loaded")

    def calc_initial_stats(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Calculating initial model statistics")
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
        self.metrics_mgr.record_pruning({
            "parameters": {"original": params},
            "flops": {"original": flops},
            "filters": {"original": filters},
            "model_size_mb": {"original": size_mb},
        })
        return self.initial_stats

    def pretrain(self, *, device: str | int | list = 0, label_fn=None, **train_kwargs: Any) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Training model to collect activations")
        train_kwargs.setdefault("plots", False)
        train_kwargs.setdefault("epochs", 1)
        if label_fn is None:
            label_fn = lambda batch: batch["cls"]
        self._register_label_callback(label_fn)
        original_model = self.model.model
        metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        self._unregister_label_callback()

        model_changed = self.model.model is not original_model
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
            if model_changed:
                try:
                    import torch
                    example_inputs = getattr(self.pruning_method, "example_inputs", None)
                    if torch.is_tensor(example_inputs):
                        device = next(self.pruning_method.model.parameters()).device
                        self.pruning_method.example_inputs = example_inputs.to(device)
                    self.pruning_method.analyze_model()
                    self.logger.debug("reanalyzed pruning method model")
                except Exception:
                    pass

        num_labels = len(getattr(self.pruning_method, "labels", []))
        self.logger.info("Training finished; recorded %d label batches", num_labels)
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["pretrain"] = metrics or {}
        return metrics or {}

    def analyze_structure(self) -> None:
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            raise NotImplementedError("PruningPipeline2 requires DepgraphHSICMethod")
        self.logger.info("Analyzing model structure")
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
        self.pruning_method.analyze_model()
        groups = len(getattr(self.pruning_method, "channel_groups", []))
        convs = len(getattr(self.pruning_method, "layers", []))
        self.logger.info(
            "Analysis summary: %d convolution layers, %d channel groups",
            convs,
            groups,
        )

    def generate_pruning_mask(self, ratio: float) -> None:
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            raise NotImplementedError
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        self.pruning_method.generate_pruning_mask(ratio)
        channels = sum(len(v) for v in getattr(self.pruning_method, "pruning_plan", {}).values())
        total = sum(
            getattr(layer, "out_channels", 0)
            for layer in getattr(self.pruning_method, "layers", [])
        )
        ratio_pruned = (channels / total * 100) if total else 0
        self.logger.info(
            "Mask summary: %d/%d channels selected for pruning (%.2f%%)",
            channels,
            total,
            ratio_pruned,
        )

    def apply_pruning(self) -> None:
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            raise NotImplementedError
        self.logger.info("Applying pruning via DependencyGraph")
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
            self.pruning_method.refresh_dependency_graph()
            self.logger.info("Dependency graph refreshed before pruning")
        self.pruning_method.apply_pruning()
        try:
            import torch_pruning as tp
            tp.utils.remove_pruning_reparametrization(self.model.model)
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.debug("remove_pruning_reparametrization failed: %s", exc)
        pruned = sum(len(v) for v in getattr(self.pruning_method, "pruning_plan", {}).values())
        self.logger.info("Pruning applied; %d channels pruned", pruned)

    def reconfigure_model(self, output_path: str | Path | None = None) -> None:
        self.logger.info("Skipping explicit reconfiguration – handled by depgraph")
        self.logger.info("Dependency graph reconfiguration complete")

    def calc_pruned_stats(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Calculating pruned model statistics")
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
        self.metrics_mgr.record_pruning({
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
            "filters": {
                "pruned": filters,
                "reduction": orig_filters - filters,
                "reduction_percent": ((orig_filters - filters) / orig_filters * 100) if orig_filters else 0,
            },
            "model_size_mb": {
                "pruned": size_mb,
                "reduction": orig_size - size_mb,
                "reduction_percent": ((orig_size - size_mb) / orig_size * 100) if orig_size else 0,
            },
        })
        log_stats_comparison(self.initial_stats, self.pruned_stats, self.logger)
        return self.pruned_stats

    def finetune(self, *, device: str | int | list = 0, label_fn=None, **train_kwargs: Any) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Finetuning pruned model")
        train_kwargs.setdefault("plots", False)
        if label_fn is None:
            label_fn = lambda batch: batch["cls"]
        self._register_label_callback(label_fn)
        metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        self._unregister_label_callback()
        self.logger.info("Finetuning completed")
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["finetune"] = metrics or {}
        return metrics or {}

__all__ = ["PruningPipeline2"]
