from __future__ import annotations

from typing import Any

from ..context import PipelineContext
from . import PipelineStep


class TrainStep(PipelineStep):
    """Train or finetune the model using ``YOLO.train``."""

    def __init__(self, phase: str, **train_kwargs: Any) -> None:
        self.phase = phase
        self.train_kwargs = train_kwargs

    def run(self, context: PipelineContext) -> None:
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Training model (%s)", self.phase)
        metrics = context.model.train(data=context.data, **self.train_kwargs)
        context.metrics[self.phase] = metrics or {}

__all__ = ["TrainStep"]
