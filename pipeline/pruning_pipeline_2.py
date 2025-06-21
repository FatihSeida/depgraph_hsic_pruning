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

    def _run_short_forward_pass(self) -> None:
        """Execute :class:`ShortForwardPassStep` to collect activations."""
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            return
        try:  # pragma: no cover - optional heavy deps
            from . import ShortForwardPassStep, PipelineContext

            ctx = PipelineContext(
                model_path=self.model_path,
                data=self.data,
                workdir=Path(self.workdir),
                pruning_method=self.pruning_method,
                logger=self.logger,
            )
            ctx.model = self.model
            ctx.metrics_mgr = self.metrics_mgr
            ShortForwardPassStep().run(ctx)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("short forward pass failed: %s", exc)

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

    def _sync_example_inputs_device(self) -> None:
        """Move ``example_inputs`` to the current model's device if needed."""
        if not (
            isinstance(self.pruning_method, DepgraphHSICMethod)
            and hasattr(self.pruning_method, "example_inputs")
            and self.model is not None
        ):
            return
        try:  # pragma: no cover - best effort
            import torch

            device = next(self.model.model.parameters()).device
            ex_inputs = getattr(self.pruning_method, "example_inputs")
            ex_device = ex_inputs.device if torch.is_tensor(ex_inputs) else None
            if torch.is_tensor(ex_inputs):
                self.pruning_method.example_inputs = ex_inputs.to(device)
                ex_device = self.pruning_method.example_inputs.device
            self.logger.debug(
                "Model device: %s, example_inputs device: %s", device, ex_device
            )
        except Exception:
            pass

    def _sync_pruning_method(self, reanalyze: bool = False) -> None:
        """Synchronize :attr:`pruning_method` with the current model."""
        if self.pruning_method is None or self.model is None:
            return
        self.pruning_method.model = self.model.model
        if reanalyze:
            self._sync_example_inputs_device()
            self.pruning_method.analyze_model()

    # ------------------------------------------------------------------
    # BasePruningPipeline interface
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        self.logger.info("Loading model from %s", self.model_path)
        self.model = YOLO(self.model_path)
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
            self._sync_example_inputs_device()
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
        try:
            metrics = self.model.train(data=self.data, device=device, **train_kwargs)
        finally:
            self._unregister_label_callback()
        model_changed = self.model.model is not original_model

        num_labels = len(getattr(self.pruning_method, "labels", []))

        if self.pruning_method is not None:
            try:
                self._sync_pruning_method(reanalyze=model_changed)
                if model_changed:
                    convs = len(getattr(self.pruning_method, "layers", []))
                    self.logger.info(
                        "Dependency graph rebuilt; %d convolution layers registered",
                        convs,
                    )
                self._run_short_forward_pass()
            except Exception:
                pass

        self.logger.info("Training finished; recorded %d label batches", num_labels)
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["pretrain"] = metrics or {}
        return metrics or {}

    def analyze_structure(self) -> None:
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            raise NotImplementedError("PruningPipeline2 requires DepgraphHSICMethod")
        self.logger.info("Analyzing model structure")
        self._sync_pruning_method(reanalyze=True)
        groups = len(getattr(self.pruning_method, "channel_groups", []))
        convs = len(getattr(self.pruning_method, "layers", []))
        self.logger.info(
            "Analysis summary: %d convolution layers, %d channel groups",
            convs,
            groups,
        )

    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            raise NotImplementedError
        self.logger.info("Generating pruning mask at ratio %.2f", ratio)
        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
            self.logger.info("Reanalyzing model before mask generation")
            self.pruning_method.analyze_model()
            if not getattr(self.pruning_method, "activations", None) or not getattr(self.pruning_method, "labels", None):
                self.logger.info("No activations/labels found; running short forward pass")
                self._run_short_forward_pass()
                if not getattr(self.pruning_method, "activations", None) or not getattr(self.pruning_method, "labels", None):
                    self.logger.warning("Short forward pass did not record activations/labels")
        if dataloader is None:
            try:
                trainer = getattr(self.model, "trainer", None)
                dataloader = getattr(trainer, "train_loader", None) or getattr(trainer, "train_dataloader", None) or getattr(trainer, "val_loader", None) or getattr(trainer, "val_dataloader", None)
            except Exception:  # pragma: no cover - best effort
                dataloader = None
        self.pruning_method.generate_pruning_mask(ratio, dataloader)
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

    def apply_pruning(self, rebuild: bool = False) -> None:
        """Apply the pruning plan using ``DependencyGraph``.

        Parameters
        ----------
        rebuild : bool, optional
            Deprecated and ignored. Maintained for backward compatibility.
        """
        if not isinstance(self.pruning_method, DepgraphHSICMethod):
            raise NotImplementedError
        self.logger.info("Applying pruning via DependencyGraph")
        if self.pruning_method is not None:
            self._sync_pruning_method(reanalyze=True)

            plan = getattr(self.pruning_method, "pruning_plan", [])
            if isinstance(plan, dict):
                named = dict(self.model.model.named_modules())
                try:
                    import torch_pruning as tp
                except Exception:
                    tp = None
                for name, idxs in plan.items():
                    layer = named.get(name)
                    if layer is None or tp is None:
                        continue
                    group = self.pruning_method.DG.get_pruning_group(
                        layer, tp.prune_conv_out_channels, idxs
                    )
                    try:
                        self.pruning_method.DG.prune_group(group)
                    except AttributeError:
                        group.prune()
            else:
                for group in plan:
                    try:
                        self.pruning_method.DG.prune_group(group)
                    except AttributeError:
                        group.prune()

            try:
                import torch_pruning as tp
                tp.utils.remove_pruning_reparametrization(self.model.model)
            except Exception:  # pragma: no cover - torch_pruning optional
                pass

        pruned = sum(len(v) for v in getattr(self.pruning_method, "pruning_plan", {}).values()) if isinstance(self.pruning_method.pruning_plan, dict) else len(getattr(self.pruning_method, "pruning_plan", []))
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

        if self.pruning_method is not None:
            self.pruning_method.model = self.model.model
            try:
                self._sync_example_inputs_device()
                self.pruning_method.analyze_model()
                convs = len(getattr(self.pruning_method, "layers", []))
                self.logger.debug(
                    "reanalyzed pruning method model; %d convolution layers registered",
                    convs,
                )
            except Exception:
                pass

        self.logger.info("Finetuning completed")
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["finetune"] = metrics or {}
        return metrics or {}

__all__ = ["PruningPipeline2"]
