from pipeline.context import PipelineContext
from pipeline.step.generate_masks import GenerateMasksStep


def test_generate_masks_step_uses_context_loader(tmp_path):
    loader = object()

    calls = []

    class DummyMethod:
        def generate_pruning_mask(self, ratio, dataloader=None):
            calls.append(dataloader)

    ctx = PipelineContext('m', 'd', workdir=tmp_path)
    ctx.pruning_method = DummyMethod()
    ctx.dataloader = loader

    step = GenerateMasksStep(ratio=0.5)
    step.run(ctx)

    assert calls == [loader]
