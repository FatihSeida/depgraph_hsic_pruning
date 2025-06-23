import os
import sys
import types
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ensure real torch is available
import torch  # noqa: F401


def test_pruning_pipeline2_model_updated_after_training(monkeypatch):
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = torch_utils

    class DummyYOLO:
        def __init__(self):
            self.model = types.SimpleNamespace(id=0)
            self.callbacks = {}
        def add_callback(self, event, cb):
            self.callbacks.setdefault(event, []).append(cb)
        def train(self, *a, **k):
            self.model = types.SimpleNamespace(id=self.model.id + 1)
            return {}

    up.YOLO = lambda *a, **k: DummyYOLO()
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')
    class DummyMethod:
        def __init__(self, model=None, **kw):
            self.model = model
            self.calls = 0
        def analyze_model(self):
            self.calls += 1
    hsic_mod.DepgraphHSICMethod = DummyMethod
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic_mod)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)
    pp.YOLO = up.YOLO

    method = DummyMethod(None)
    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=method)
    pipeline.model = DummyYOLO()
    pipeline.pretrain()
    assert pipeline.pruning_method.model is pipeline.model.model
    assert method.calls == 1
    pipeline.analyze_structure()
    assert method.calls == 2
    pipeline.finetune()
    assert pipeline.pruning_method.model is pipeline.model.model
    assert method.calls == 3
    pipeline.analyze_structure()
    assert method.calls == 4
