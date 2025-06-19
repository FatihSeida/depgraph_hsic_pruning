from __future__ import annotations

from typing import Any, Callable


from ..context import PipelineContext
from . import PipelineStep


class TrainStep(PipelineStep):
    """Train or finetune the model using ``YOLO.train``.

    Parameters passed via ``train_kwargs`` are forwarded to ``YOLO.train``. By
    default plotting is enabled unless ``plots=False`` is specified.
    """

    def __init__(self, phase: str, label_fn: Callable[[dict], Any] | None = None, **train_kwargs: Any) -> None:
        self.phase = phase
        self.label_fn = label_fn or (lambda batch: batch["cls"])
        self.train_kwargs = train_kwargs

    def run(self, context: PipelineContext) -> None:
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Training model (%s)", self.phase)
        self.train_kwargs.setdefault("plots", True)

        # Automatically record labels after each forward pass when using
        # ``DepgraphHSICMethod``. ``on_train_batch_end`` runs once per batch
        # after the model has produced outputs, ensuring activations and labels
        # stay aligned.  Avoid registering duplicate callbacks across multiple
        # training phases.
        try:
            from prune_methods.depgraph_hsic import DepgraphHSICMethod  # local import to avoid heavy dependency at module import
        except Exception:  # pragma: no cover - dependency missing
            DepgraphHSICMethod = None

        if DepgraphHSICMethod is not None and isinstance(getattr(context, "pruning_method", None), DepgraphHSICMethod):
            def record_labels(trainer) -> None:  # pragma: no cover - heavy dependency
                batch = getattr(trainer, "batch", None)
                if isinstance(batch, dict) and "cls" in batch:
                    labels = self.label_fn(batch)
                    try:
                        import torch  # local import to avoid hard dependency at module import
                        if torch.is_tensor(labels) and len(labels) != batch["img"].shape[0]:
                            context.logger.warning(
                                "label_fn returned %d labels for batch size %d; "
                                "labels are likely object-level and may cause activation/label mismatches",
                                len(labels),
                                batch["img"].shape[0],
                            )
                    except Exception:  # pragma: no cover - if torch missing or labels malformed
                        pass
                    context.logger.debug(
                        "Adding labels for batch with shape %s", tuple(labels.shape)
                    )
                    context.pruning_method.add_labels(labels)

            try:
                existing = getattr(context.model, "callbacks", {}).get("on_train_batch_end", [])
                if record_labels not in existing:
                    context.model.add_callback("on_train_batch_end", record_labels)
            except AttributeError:  # pragma: no cover - fallback for stubs
                pass

        original_model = getattr(context.model, "model", None)
        metrics = context.model.train(data=context.data, **self.train_kwargs)
        model_changed = getattr(context.model, "model", None) is not original_model
        pm = getattr(context, "pruning_method", None)
        if model_changed and pm is not None:
            try:
                pm.model = context.model.model
                context.logger.debug("updated pruning method model reference")
            except Exception:  # pragma: no cover - best effort
                pass
            try:
                if hasattr(pm, "refresh_dependency_graph"):
                    # Avoid clearing collected activations when the model instance changes
                    pm.refresh_dependency_graph()
                    context.logger.debug("refreshed pruning method dependency graph")
                else:
                    import torch  # local import to avoid hard dependency at module import
                    try:
                        device = next(pm.model.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")
                    if torch.is_tensor(pm.example_inputs):
                        pm.example_inputs = pm.example_inputs.to(device)
                    pm.analyze_model()
                    context.logger.debug("reanalyzed pruning method model")
            except Exception:  # pragma: no cover - best effort
                pass
        context.metrics_mgr.record_training(metrics or {})
        context.metrics[self.phase] = metrics or {}
        context.logger.info("Finished %s", step)

__all__ = ["TrainStep"]
