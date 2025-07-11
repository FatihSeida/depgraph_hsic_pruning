import importlib
import sys
import types
import torch
import pytest


def setup(monkeypatch):
    tp = types.ModuleType('torch_pruning')

    class DummyDG:
        def build_dependency(self, model, example_inputs=None):
            pass
        def get_all_groups(self, root_module_types=None):
            return []

    class DummyPruner:
        def __init__(self, *a, **k):
            pass
        def step(self):
            pass

    tp.DependencyGraph = DummyDG
    tp.importance = types.SimpleNamespace(
        MagnitudeImportance=lambda *a, **k: None,
        RandomImportance=lambda *a, **k: None,
        GroupMagnitudeImportance=lambda *a, **k: None,
    )
    tp.MagnitudePruner = DummyPruner
    tp.RandomPruner = DummyPruner
    tp.pruner = types.SimpleNamespace(
        algorithms=types.SimpleNamespace(BasePruner=DummyPruner, GroupNormPruner=DummyPruner)
    )
    tp.utils = types.SimpleNamespace(remove_pruning_reparametrization=lambda m: None)
    monkeypatch.setitem(sys.modules, 'torch_pruning', tp)

    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    tu = types.ModuleType('ultralytics.utils.torch_utils')
    tu.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = tu
    utils.DEFAULT_CFG = types.SimpleNamespace(batch=1, workers=0, imgsz=32)
    utils.yaml_load = lambda f, append_filename=False: {"path": ".", "val": "images"}

    cfg_mod = types.ModuleType('ultralytics.cfg')
    cfg_mod.get_cfg = lambda cfg=None: types.SimpleNamespace(batch=1, workers=0, imgsz=32, data='d')

    data_mod = types.ModuleType('ultralytics.data')
    data_mod.build_yolo_dataset = lambda *a, **k: 'ds'
    data_mod.build_dataloader = lambda *a, **k: object()

    class DummyYOLO:
        def __init__(self):
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 8, 3),
                torch.nn.ReLU(),
            )
            self.callbacks = {}
            self.trainer = types.SimpleNamespace(val_loader=None)
        def add_callback(self, event, cb):
            pass

    up.YOLO = lambda *a, **k: DummyYOLO()
    up.utils = utils
    up.cfg = cfg_mod
    up.data = data_mod
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', tu)
    monkeypatch.setitem(sys.modules, 'ultralytics.cfg', cfg_mod)
    monkeypatch.setitem(sys.modules, 'ultralytics.data', data_mod)
    monkeypatch.setitem(sys.modules, 'ultralytics.data.loaders', data_mod)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)
    dep_mod = importlib.import_module('prune_methods.depgraph_pruning')
    importlib.reload(dep_mod)
    rand_mod = importlib.import_module('prune_methods.torch_pruning_simple')
    importlib.reload(rand_mod)
    return pp, dep_mod.DepgraphMethod, rand_mod.TorchRandomMethod


def test_pipeline2_depgraph_and_random_methods(monkeypatch):
    pp, DepgraphMethod, TorchRandomMethod = setup(monkeypatch)

    for Method in (DepgraphMethod, TorchRandomMethod):
        method = Method(None)
        pipeline = pp.PruningPipeline2('m', 'd', pruning_method=method)
        pipeline.load_model()
        pipeline.analyze_structure()
        pipeline.generate_pruning_mask(0.5)
