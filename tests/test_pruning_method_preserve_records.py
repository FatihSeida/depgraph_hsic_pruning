import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ensure real torch is available
import torch  # noqa: F401

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_num_params = lambda *a, **k: 0
from helper import flops_utils as fu
fu.get_flops_reliable = lambda *a, **k: 0
utils.torch_utils = torch_utils

class DummyYOLO:
    def __init__(self):
        self.model = types.SimpleNamespace(id=0)
        self.callbacks = {}
    def add_callback(self, event, cb):
        self.callbacks.setdefault(event, []).append(cb)
    def train(self, *a, **k):
        # mutate model but keep same instance
        self.model.id += 1
        return {}

up.YOLO = lambda *a, **k: DummyYOLO()
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

base_mod = types.ModuleType('prune_methods.base')
class DummyMethod:
    def __init__(self, model=None, **kw):
        self.model = model
        self.calls = 0
        self.activations = {0: [0]}
        self.layer_shapes = {0: (1, 1)}
        self.num_activations = {0: 1}
        self.labels = [0]
    def analyze_model(self):
        self.calls += 1
        # emulate analyze_model clearing records
        self.activations = {}
        self.layer_shapes = {}
        self.num_activations = {}
        self.labels = []
base_mod.BasePruningMethod = DummyMethod
sys.modules['prune_methods.base'] = base_mod

import importlib
pp = importlib.import_module('pipeline.pruning_pipeline')
pp.YOLO = up.YOLO


def test_records_preserved_when_model_unchanged():
    method = DummyMethod(None)
    pipeline = pp.PruningPipeline('m', 'd', pruning_method=method)
    pipeline.model = DummyYOLO()
    pipeline.pretrain()
    assert method.calls == 0
    pipeline.analyze_structure()
    assert method.calls == 1
    assert method.activations == {}
    pipeline.finetune()
    assert method.calls == 1
    pipeline.analyze_structure()
    assert method.calls == 2
