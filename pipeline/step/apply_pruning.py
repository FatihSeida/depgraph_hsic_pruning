from __future__ import annotations

from ..context import PipelineContext
from . import PipelineStep


class ApplyPruningStep(PipelineStep):
    """Apply the generated pruning mask to the model."""

    def run(self, context: PipelineContext) -> None:
        step = self.__class__.__name__
        context.logger.info("Starting %s", step)
        if context.pruning_method is None:
            raise NotImplementedError
        context.logger.info("Applying pruning mask")

        # sync and prune directly without using the method's apply_pruning
        try:
            context.pruning_method.model = context.model.model
            context.pruning_method.analyze_model()
        except Exception:  # pragma: no cover - best effort
            pass

        plan = getattr(context.pruning_method, "pruning_plan", [])
        if isinstance(plan, dict):
            named = dict(context.model.model.named_modules())
            try:
                import torch_pruning as tp
            except Exception:
                tp = None
            for name, idxs in plan.items():
                layer = named.get(name)
                if layer is None or tp is None:
                    continue
                group = context.pruning_method.DG.get_pruning_group(
                    layer, tp.prune_conv_out_channels, idxs
                )
                try:
                    context.pruning_method.DG.prune_group(group)
                except AttributeError:
                    group.prune()
        else:
            for group in plan:
                try:
                    context.pruning_method.DG.prune_group(group)
                except AttributeError:
                    group.prune()

        try:
            import torch_pruning as tp
            tp.utils.remove_pruning_reparametrization(context.model.model)
        except Exception:  # pragma: no cover - optional dependency
            pass
        snapshot = context.workdir / "snapshot.pt"
        try:
            context.logger.info("Saving snapshot to %s", snapshot)
            context.model.save(str(snapshot))
        except Exception:  # pragma: no cover - best effort
            pass
        context.logger.info("Finished %s", step)

__all__ = ["ApplyPruningStep"]
