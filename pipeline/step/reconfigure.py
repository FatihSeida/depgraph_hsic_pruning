from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep
from ..model_reconfig import AdaptiveLayerReconfiguration


class ReconfigureModelStep(PipelineStep):
    """Reconfigure pruned model layers to match pruned channels."""

    def __init__(self) -> None:
        self.reconfigurator = AdaptiveLayerReconfiguration()

    def run(self, context: PipelineContext) -> None:
        if context.model is None:
            raise ValueError("Model is not loaded")
        context.logger.info("Reconfiguring model")
        self.reconfigurator.reconfigure_model(context.model)

__all__ = ["ReconfigureModelStep"]
