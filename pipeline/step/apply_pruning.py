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

        # sync method with the current model and rebuild the dependency graph
        try:
            context.pruning_method.model = context.model.model
            ex_inputs = getattr(context.pruning_method, "example_inputs", None)
            if ex_inputs is not None:
                try:
                    import torch

                    device = next(context.model.model.parameters()).device
                    if torch.is_tensor(ex_inputs):
                        context.pruning_method.example_inputs = ex_inputs.to(device)
                except Exception:  # pragma: no cover - optional deps missing
                    pass
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
        try:
            context.pruning_method.remove_hooks()
        except Exception:
            pass
        snapshot = context.workdir / "snapshot.pt"
        try:
            context.logger.info("Saving snapshot to %s", snapshot)
            context.model.save(str(snapshot))
        except Exception:  # pragma: no cover - best effort
            pass
        context.logger.info("Finished %s", step)

__all__ = ["ApplyPruningStep"]
