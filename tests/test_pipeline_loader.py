import importlib
import sys
import types


def setup(monkeypatch, loader):
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_flops = lambda *a, **k: 0
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils

    class DummyYOLO:
        def __init__(self):
            self.model = types.SimpleNamespace()
            self.callbacks = {}
            self.trainer = types.SimpleNamespace(val_loader=loader)
        def add_callback(self, event, cb):
            pass
    up.utils = utils
    up.YOLO = lambda *a, **k: DummyYOLO()
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')
    class Base:
        def __init__(self, model=None, **kw):
            self.model = model
    hsic_mod.DepgraphHSICMethod = Base
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic_mod)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)
    return pp, hsic_mod.DepgraphHSICMethod


def test_loader_passed(monkeypatch):
    loader = object()
    pp, Base = setup(monkeypatch, loader)

    calls = []

    class DummyMethod(Base):
        def __init__(self, model=None, **kw):
            super().__init__(model)
        def analyze_model(self):
            pass
        def generate_pruning_mask(self, ratio, dataloader=None):
            calls.append((dataloader,))
        def apply_pruning(self):
            pass

    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=DummyMethod(None))
    pipeline.load_model()
    pipeline.generate_pruning_mask(0.5)
    assert calls == [(loader,)]
