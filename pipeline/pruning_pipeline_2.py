from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

from ultralytics import YOLO
from helper.flops_utils import get_flops_reliable, get_num_params_reliable
from torch import nn

from .base_pipeline import BasePruningPipeline
from prune_methods.depgraph_hsic import DepgraphHSICMethod
from prune_methods.base import BasePruningMethod

from helper import (
    Logger,
    MetricManager,
    count_filters,
    model_size_mb,
    log_stats_comparison,
    format_training_summary,
)


class PruningPipeline2(BasePruningPipeline):
    """Simple pipeline specialised for DepGraph pruning methods.

    The pipeline works with any :class:`BasePruningMethod` that relies on the
    ``torch-pruning`` dependency graph.  Additional HSIC specific logic is only
    executed when the pruning method is an instance of
    :class:`DepgraphHSICMethod`.
    """

    def __init__(
        self,
        model_path: str,
        data: str,
        workdir: str = "runs/pruning",

        pruning_method: BasePruningMethod | None = None,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(model_path, data, workdir, pruning_method, logger)
        self.model: YOLO | None = None
        self.metrics_mgr = MetricManager()
        self.metrics_csv: Path | None = None

    def _is_depgraph_method(self) -> bool:
        try:
            from prune_methods.depgraph_hsic import DepgraphHSICMethod
        except Exception:
            return False
        return isinstance(self.pruning_method, DepgraphHSICMethod)

    def _sync_example_inputs_device(self) -> None:
        """Move ``example_inputs`` to the current model's device if needed."""
        if not (
            self._is_depgraph_method()
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
        params = get_num_params_reliable(self.model.model)
        flops = get_flops_reliable(self.model.model)
        if flops == 0:
            self.logger.warning(
                "FLOPs reported as 0; fallback calculation may be inaccurate"
            )
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
            except Exception:
                self.logger.exception("failed to refresh pruning method")

        self.logger.info("Training finished; recorded %d label batches", num_labels)
        if metrics:
            self.logger.info("Training summary: %s", format_training_summary(metrics))
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["pretrain"] = metrics or {}
        return metrics or {}

    def analyze_structure(self) -> None:
        if self.pruning_method is None:
            raise ValueError("No pruning method set")
        self._sync_pruning_method(reanalyze=True)
        groups = []
        if getattr(self.pruning_method, "DG", None) is not None:
            try:
                groups = list(
                    self.pruning_method.DG.get_all_groups(
                        root_module_types=(nn.Conv2d,)
                    )
                )
            except Exception:
                groups = []
        convs = len(getattr(self.pruning_method, "layers", []))
        self.logger.info(
            "Analysis summary: %d convolution layers, %d dependency groups",
            convs,
            len(groups),
        )

    def _build_val_loader(self):
        """Return a validation dataloader built from ``self.data``."""
        if self.model is None:
            raise ValueError("Model is not loaded")
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG, YAML
        from ultralytics.data import build_yolo_dataset, build_dataloader

        cfg = get_cfg(DEFAULT_CFG)
        cfg.data = self.data

        data_dict = YAML.load(self.data)
        img_path = data_dict.get("val") or data_dict.get("test") or data_dict.get("train")
        stride = int(max(getattr(self.model.model, "stride", [32])))

        dataset = build_yolo_dataset(cfg, img_path, cfg.batch, data_dict, mode="val", rect=True, stride=stride)
        return build_dataloader(dataset, batch=cfg.batch, workers=cfg.workers * 2, shuffle=False, rank=-1)

    def generate_pruning_mask(
        self,
        ratio: float,
        dataloader: Any | None = None,
    ) -> None:
        if self.pruning_method is None:
            raise ValueError("No pruning method set")
        self.pruning_method.model = self.model.model
        self.logger.info("Reanalyzing model before mask generation")
        self.pruning_method.analyze_model()
        if dataloader is None:
            dataloader = getattr(getattr(self.model, "trainer", None), "val_loader", None)
            if dataloader is None:
                try:
                    dataloader = self._build_val_loader()
                except Exception as exc:
                    raise ValueError("failed to build validation dataloader") from exc

        if isinstance(self.pruning_method, DepgraphHSICMethod):
            if dataloader is None:
                raise ValueError("dataloader is required")
            self.pruning_method.generate_pruning_mask(
                ratio,
                dataloader=dataloader,
            )
        else:
            self.pruning_method.generate_pruning_mask(
                ratio,
                dataloader=dataloader,
            )
        plan = getattr(self.pruning_method, "pruning_plan", [])
        channels = len(plan)
        total = sum(
            getattr(layer, "out_channels", 0)
            for layer in getattr(self.pruning_method, "layers", [])
        )
        ratio_pruned = (channels / total * 100) if total else 0
        self.logger.info(
            "Mask summary: %d groups selected for pruning (%.2f%% of %d channels)",
            channels,
            ratio_pruned,
            total,
        )

    def apply_pruning(self, rebuild: bool = False) -> None:
        if self.pruning_method is None:
            raise ValueError("No pruning method set")
        import inspect
        try:
            sig = inspect.signature(self.pruning_method.apply_pruning)
            if 'rebuild' in sig.parameters:
                self.pruning_method.apply_pruning(rebuild=rebuild)
            else:
                self.pruning_method.apply_pruning()
        except (ValueError, TypeError):  # pragma: no cover - fallback
            self.pruning_method.apply_pruning(rebuild=rebuild)

    def reconfigure_model(self, output_path: str | Path | None = None) -> None:
        # DepGraph methods don't require reconfiguration
        self.logger.info("Skipping reconfiguration (not required for DepGraph methods)")

    def calc_pruned_stats(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Calculating pruned model statistics")
        params = get_num_params_reliable(self.model.model)
        flops = get_flops_reliable(self.model.model)
        if flops == 0:
            self.logger.warning(
                "FLOPs reported as 0; fallback calculation may be inaccurate"
            )
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
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.logger.info("Finetuning pruned model")
        train_kwargs.setdefault("plots", False)
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
            try:
                self._sync_pruning_method(reanalyze=model_changed)
            except Exception:
                self.logger.exception("failed to sync pruning method")
        if metrics:
            self.logger.debug("Training summary: %s", format_training_summary(metrics))
            self.logger.info("Training summary: %s", format_training_summary(metrics))
        self.metrics_mgr.record_training(metrics or {})
        self.metrics["finetune"] = metrics or {}
        return metrics or {}

__all__ = ["PruningPipeline2"]
