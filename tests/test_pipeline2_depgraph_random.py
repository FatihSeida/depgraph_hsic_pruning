import importlib
import sys
import types
import torch


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
    tu.get_flops = lambda *a, **k: 0
    tu.get_num_params = lambda *a, **k: 0
    utils.torch_utils = tu

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
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', tu)

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
        pipeline.apply_pruning()
