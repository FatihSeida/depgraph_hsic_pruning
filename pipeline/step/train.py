from __future__ import annotations

from typing import Any

from ..context import PipelineContext
from . import PipelineStep


class TrainStep(PipelineStep):
    """Train or finetune the model using ``YOLO.train``.

    Parameters passed via ``train_kwargs`` are forwarded to ``YOLO.train``. By
    default plotting is enabled unless ``plots=False`` is specified.
    """

    def __init__(self, phase: str, **train_kwargs: Any) -> None:
        self.phase = phase
        self.train_kwargs = train_kwargs

    def run(self, context: PipelineContext) -> None:
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Training model (%s)", self.phase)
        self.train_kwargs.setdefault("plots", True)

        # Automatically record labels after each forward pass when using
        # ``DepgraphHSICMethod``. ``on_train_batch_end`` runs once per batch
        # after the model has produced outputs, ensuring activations and labels
        # stay aligned.  Avoid registering duplicate callbacks across multiple
        # training phases.
        if getattr(context, "pruning_method", None).__class__.__name__ == "DepgraphHSICMethod":
            def record_labels(trainer) -> None:  # pragma: no cover - heavy dependency
                batch = getattr(trainer, "batch", None)
                if isinstance(batch, dict) and "cls" in batch:
                    context.pruning_method.add_labels(batch["cls"])

            try:
                existing = getattr(context.model, "callbacks", {}).get("on_train_batch_end", [])
                if record_labels not in existing:
                    context.model.add_callback("on_train_batch_end", record_labels)
            except AttributeError:  # pragma: no cover - fallback for stubs
                pass

        metrics = context.model.train(data=context.data, **self.train_kwargs)
        context.metrics_mgr.record_training(metrics or {})
        context.metrics[self.phase] = metrics or {}

__all__ = ["TrainStep"]
