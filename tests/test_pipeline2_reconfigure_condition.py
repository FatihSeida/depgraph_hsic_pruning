import importlib
import sys
import types


def setup(monkeypatch):
    up = types.ModuleType("ultralytics")
    utils = types.ModuleType("ultralytics.utils")
    tu = types.ModuleType("ultralytics.utils.torch_utils")
    tu.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu

    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = tu
    up.utils = utils
    up.YOLO = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ultralytics", up)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", tu)

    pp = importlib.import_module("pipeline.pruning_pipeline_2")
    importlib.reload(pp)
    return pp


def test_reconfigure_called_when_required(monkeypatch):
    pp = setup(monkeypatch)

    calls = []

    class DummyReconfig:
        def __init__(self, logger=None):
            pass

        def reconfigure_model(self, model, output_path=None):
            calls.append((model, output_path))

    monkeypatch.setattr(pp, "AdaptiveLayerReconfiguration", DummyReconfig)

    class DummyMethod:
        requires_reconfiguration = True

        def __init__(self, model=None, **kw):
            self.model = model

        def analyze_model(self):
            pass

        def generate_pruning_mask(self, ratio, dataloader=None):
            pass

        def apply_pruning(self, rebuild=False):
            pass

    pipeline = pp.PruningPipeline2("m", "d", pruning_method=DummyMethod(None))
    pipeline.model = types.SimpleNamespace(model=object())
    pipeline.reconfigure_model("out.pt")

    assert calls == [(pipeline.model, "out.pt")]
