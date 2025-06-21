import importlib
import sys
import types

import pytest


def test_apply_pruning_uses_method(monkeypatch):
    tp = types.ModuleType('torch_pruning')
    tp.utils = types.SimpleNamespace(remove_pruning_reparametrization=lambda m: None)
    monkeypatch.setitem(sys.modules, 'torch_pruning', tp)

    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_flops = lambda *a, **k: 0
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    up.utils = utils
    up.YOLO = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)

    calls = []

    class DummyMethod:
        def __init__(self, model=None, **kw):
            self.model = model
            self.pruning_plan = [object()]
        def apply_pruning(self):
            calls.append(self.model)

    monkeypatch.setattr(pp, 'DepgraphHSICMethod', DummyMethod)

    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=DummyMethod(None))
    pipeline.model = types.SimpleNamespace(model=object())

    pipeline.apply_pruning()

    assert calls == [pipeline.model.model]
