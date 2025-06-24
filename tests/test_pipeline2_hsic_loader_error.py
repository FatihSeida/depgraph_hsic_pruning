import importlib
import sys
import types
import pytest


def test_pipeline2_hsic_requires_loader(monkeypatch):
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    tu = types.ModuleType('ultralytics.utils.torch_utils')
    tu.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = tu

    class DummyYOLO:
        def __init__(self):
            self.model = types.SimpleNamespace()
            self.callbacks = {}
            self.trainer = types.SimpleNamespace(val_loader=None)
        def add_callback(self, event, cb):
            pass

    up.utils = utils
    up.YOLO = lambda *a, **k: DummyYOLO()
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', tu)

    hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')

    class DummyMethod:
        def __init__(self, model=None, **kw):
            self.model = model
            self.called = False
        def analyze_model(self):
            pass
        def generate_pruning_mask(self, ratio, dataloader=None):
            self.called = True

    hsic_mod.DepgraphHSICMethod = DummyMethod
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic_mod)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)

    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=DummyMethod(None))
    pipeline.load_model()

    with pytest.raises(ValueError):
        pipeline.generate_pruning_mask(0.5)
    assert not pipeline.pruning_method.called
