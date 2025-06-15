from __future__ import annotations

from ultralytics import YOLO

from ..context import PipelineContext
from . import PipelineStep


class LoadModelStep(PipelineStep):
    """Load a YOLO model from ``context.model_path``."""

    def run(self, context: PipelineContext) -> None:
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        context.logger.info("Loading model from %s", context.model_path)
        context.model = YOLO(context.model_path)
        context.logger.info("Finished %s", step)

__all__ = ["LoadModelStep"]
