import importlib
import sys
import types

import pytest


class DummyTensor:
    def __init__(self, device="cpu"):
        self.device = device
    def to(self, device):
        self.device = device
        return self
    def __len__(self):
        return 1


def setup_modules(monkeypatch):
    torch = types.ModuleType('torch')
    torch.is_tensor = lambda t: isinstance(t, DummyTensor)
    torch.randn = lambda *a, **k: DummyTensor()
    monkeypatch.setitem(sys.modules, 'torch', torch)

    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = torch_utils

    class DummyModel:
        def __init__(self, device='cpu'):
            self._device = device
        def parameters(self):
            return iter([types.SimpleNamespace(device=self._device)])

    class DummyYOLO:
        def __init__(self, device='cpu'):
            self.model = DummyModel(device)
            self.callbacks = {}
        def add_callback(self, event, cb):
            pass
        def train(self, *a, **k):
            self.model = DummyModel('cuda')
            return {}

    up.utils = utils
    up.YOLO = lambda *a, **k: DummyYOLO()
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    hsic = types.ModuleType('prune_methods.depgraph_hsic')
    class DummyMethod:
        def __init__(self, model=None, **kw):
            self.model = model
            self.example_inputs = DummyTensor('cpu')
        def refresh_dependency_graph(self):
            pass
        def analyze_model(self):
            pass
        def apply_pruning(self):
            pass
    hsic.DepgraphHSICMethod = DummyMethod
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic)

    pp = importlib.import_module('pipeline.pruning_pipeline_2')
    importlib.reload(pp)
    return pp, DummyMethod


def test_example_inputs_synced_after_training(monkeypatch):
    pp, DummyMethod = setup_modules(monkeypatch)
    pipeline = pp.PruningPipeline2('m', 'd', pruning_method=DummyMethod(None))
    pipeline.load_model()
    assert pipeline.pruning_method.example_inputs.device == 'cpu'
    pipeline.pretrain()
    assert pipeline.pruning_method.example_inputs.device == 'cuda'
    pipeline.model = type(pipeline.model)()
    pipeline._sync_example_inputs_device()
    assert pipeline.pruning_method.example_inputs.device == 'cpu'

