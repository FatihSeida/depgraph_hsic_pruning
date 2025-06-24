import importlib
import sys
import types
from pathlib import Path


def test_pipeline2_uses_auto_loader_when_no_baseline(monkeypatch, tmp_path):
    loader = object()

    # stub ultralytics modules
    up = types.ModuleType('ultralytics')

    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    utils.DEFAULT_CFG = types.SimpleNamespace(batch=1, workers=0, imgsz=32)

    class DummyYAML:
        @staticmethod
        def load(file, append_filename=False):
            return {"path": str(Path(file).parent), "val": "images"}

    utils.YAML = DummyYAML
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)

    cfg_mod = types.ModuleType('ultralytics.cfg')
    cfg_mod.get_cfg = lambda cfg=None: types.SimpleNamespace(batch=1, workers=0, imgsz=32, data='d')

    data_mod = types.ModuleType('ultralytics.data')
    data_mod.build_yolo_dataset = lambda *a, **k: 'dataset'
    data_mod.build_dataloader = lambda *a, **k: loader

    up.utils = utils
    up.data = data_mod
    up.cfg = cfg_mod
    up.YOLO = lambda *a, **k: types.SimpleNamespace(
        model=types.SimpleNamespace(stride=[32]),
        callbacks={},
        trainer=None,
    )

    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.cfg', cfg_mod)
    monkeypatch.setitem(sys.modules, 'ultralytics.data', data_mod)
    monkeypatch.setitem(sys.modules, 'ultralytics.data.loaders', data_mod)

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

    pipeline = pp.PruningPipeline2('m', 'd.yaml', pruning_method=DummyMethod(None))
    pipeline.load_model()
    pipeline.analyze_structure()
    pipeline.generate_pruning_mask(0.5)
    assert pipeline.pruning_method.calls == [loader]
