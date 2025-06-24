from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class ApplyPruningStep(PipelineStep):
    """Apply the generated pruning mask to the model."""

    def run(self, context: PipelineContext) -> None:
        if context.pruning_method is None:
            raise NotImplementedError

        context.pruning_method.model = context.model.model
        context.pruning_method.apply_pruning()

        # Clean up pruning method before saving to avoid pickle issues
        if hasattr(context.pruning_method, 'cleanup_for_saving'):
            context.pruning_method.cleanup_for_saving()

        snapshot = context.workdir / "snapshot.pt"
        try:
            context.logger.info("Saving snapshot to %s", snapshot)
            context.model.save(str(snapshot))
        except Exception:  # pragma: no cover - best effort
            context.logger.exception("failed to save snapshot")

__all__ = ["ApplyPruningStep"]
