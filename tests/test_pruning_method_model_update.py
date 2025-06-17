import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ensure real torch is available
import torch  # noqa: F401

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda *a, **k: 0
torch_utils.get_num_params = lambda *a, **k: 0
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
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

base_mod = types.ModuleType('prune_methods.base')
class DummyMethod:
    def __init__(self, model=None, **kw):
        self.model = model
base_mod.BasePruningMethod = DummyMethod
sys.modules['prune_methods.base'] = base_mod

import importlib
pp = importlib.import_module('pipeline.pruning_pipeline')
pp.YOLO = up.YOLO


def test_pruning_method_model_updated_after_training():
    method = DummyMethod(None)
    pipeline = pp.PruningPipeline('m', 'd', pruning_method=method)
    pipeline.model = DummyYOLO()
    pipeline.pretrain()
    assert pipeline.pruning_method.model is pipeline.model.model
    pipeline.finetune()
    assert pipeline.pruning_method.model is pipeline.model.model
