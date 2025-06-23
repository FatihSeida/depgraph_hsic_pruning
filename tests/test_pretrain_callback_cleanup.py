import os
import sys
import types
import importlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_pretrain_unregisters_callback_on_error(monkeypatch):
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = torch_utils

    class DummyYOLO:
        def __init__(self):
            self.model = object()
            self.callbacks = {}
        def add_callback(self, event, cb):
            self.callbacks.setdefault(event, []).append(cb)
        def train(self, *a, **k):
            raise RuntimeError('boom')

    up.YOLO = lambda *a, **k: DummyYOLO()
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')
    class DummyMethod:
        def __init__(self, model=None, **kw):
            self.model = model
        def analyze_model(self):
            pass
    hsic_mod.DepgraphHSICMethod = DummyMethod
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic_mod)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)
    pp.YOLO = up.YOLO

    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=DummyMethod(None))
    pipeline.model = DummyYOLO()
    with pytest.raises(RuntimeError):
        pipeline.pretrain()

    assert pipeline._label_callback is None
    assert pipeline.model.callbacks.get('on_train_batch_end') == []
