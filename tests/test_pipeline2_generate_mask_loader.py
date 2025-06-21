import importlib
import sys
import types


def test_pipeline2_generate_mask_with_loader(monkeypatch):
    loader = object()

    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_flops = lambda *a, **k: 0
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    up.utils = utils
    up.YOLO = lambda *a, **k: types.SimpleNamespace(
        model=types.SimpleNamespace(),
        callbacks={},
        trainer=types.SimpleNamespace(val_loader=loader),
    )
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')

    class DummyMethod:
        def __init__(self, model=None, **kw):
            self.model = model
            self.calls = []

        def analyze_model(self):
            pass

        def generate_pruning_mask(self, ratio, dataloader=None):
            self.calls.append(dataloader)

    hsic_mod.DepgraphHSICMethod = DummyMethod
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic_mod)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)

    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=DummyMethod(None))
    pipeline.load_model()
    pipeline._run_short_forward_pass = lambda: (_ for _ in ()).throw(RuntimeError('called'))
    pipeline.generate_pruning_mask(0.5, dataloader=loader)
    assert pipeline.pruning_method.calls == [loader]
