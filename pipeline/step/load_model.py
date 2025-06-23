from __future__ import annotations

from ultralytics import YOLO

from ..context import PipelineContext
from . import PipelineStep


class LoadModelStep(PipelineStep):
    """Load a YOLO model from ``context.model_path``."""

    def run(self, context: PipelineContext) -> None:
        context.logger.info("Loading model from %s", context.model_path)
        context.model = YOLO(context.model_path)
        # Automatically attach the YOLO model to the pruning method if it was
        # instantiated without one.  This mirrors the quick-start example in the
        # README where ``DepgraphHSICMethod(None)`` is provided and the model is
        # assigned after loading.
        pm = getattr(context, "pruning_method", None)
        if pm is not None and getattr(pm, "model", None) is None:
            try:
                pm.model = context.model.model
                context.logger.debug(
                    "Assigned loaded model to pruning method %s", pm.__class__.__name__
                )
            except Exception:  # pragma: no cover - best effort
                pass

__all__ = ["LoadModelStep"]
