from __future__ import annotations

from typing import Any

from ..context import PipelineContext
from . import PipelineStep


class TrainStep(PipelineStep):
    """Train or finetune the model using ``YOLO.train``.

    Parameters passed via ``train_kwargs`` are forwarded to ``YOLO.train``. By
    default plotting is disabled unless ``plots=True`` is specified.
    """

    def __init__(self, phase: str, **train_kwargs: Any) -> None:
        self.phase = phase
        self.train_kwargs = train_kwargs

    def run(self, context: PipelineContext) -> None:
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Training model (%s)", self.phase)
        self.train_kwargs.setdefault("plots", False)
        metrics = context.model.train(data=context.data, **self.train_kwargs)
        context.metrics_mgr.record_training(metrics or {})
        context.metrics[self.phase] = metrics or {}

__all__ = ["TrainStep"]
