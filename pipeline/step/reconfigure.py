from __future__ import annotations

from pathlib import Path

from ..context import PipelineContext
from . import PipelineStep
from ..model_reconfig import AdaptiveLayerReconfiguration
from ultralytics import YOLO


class ReconfigureModelStep(PipelineStep):
    """Reconfigure pruned model layers to match pruned channels."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        self.reconfigurator = AdaptiveLayerReconfiguration()
        self.output_path = output_path

    def run(self, context: PipelineContext) -> None:
        if context.model is None:
            raise ValueError("Model is not loaded")
        snapshot = context.workdir / "snapshot.pt"
        if snapshot.exists():
            context.logger.info("Loading snapshot from %s", snapshot)
            context.model = YOLO(str(snapshot))
            pm = getattr(context, "pruning_method", None)
            if pm is not None:
                try:
                    pm.model = context.model.model
                except Exception:  # pragma: no cover - best effort
                    context.logger.exception("failed to assign model during reconfigure")
        context.logger.info("Reconfiguring model")
        self.reconfigurator.reconfigure_model(context.model, output_path=self.output_path)

__all__ = ["ReconfigureModelStep"]
