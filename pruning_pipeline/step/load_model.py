from __future__ import annotations

from ultralytics_pruning import YOLO

from ..context import PipelineContext
from . import PipelineStep


class LoadModelStep(PipelineStep):
    """Load a YOLO model from ``context.model_path``."""

    def run(self, context: PipelineContext) -> None:
        context.logger.info("Loading model from %s", context.model_path)
        context.model = YOLO(context.model_path)

__all__ = ["LoadModelStep"]
