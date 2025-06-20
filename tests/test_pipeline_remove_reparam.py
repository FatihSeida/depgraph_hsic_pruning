import subprocess, sys, types
from pathlib import Path

def test_pipeline_remove_pruning_reparam_called_once(tmp_path):
    code = f"""
import torch, types, sys

sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

class DummyGroup(list):
    def __init__(self, conv, idxs):
        super().__init__([(types.SimpleNamespace(target=types.SimpleNamespace(module=conv)), idxs)])
        self.conv = conv
    def prune(self):
        self.conv.reparam = True

class DummyDG:
    def build_dependency(self, model, example_inputs=None):
        pass
    def get_pruning_group(self, conv, fn, idxs):
        return DummyGroup(conv, idxs)

tp = types.ModuleType('torch_pruning')
tp.DependencyGraph = DummyDG
tp.prune_conv_out_channels = lambda *a, **k: None
calls = []

def remove_pruning_reparametrization(model):
    calls.append(model)
    for m in model.modules():
        if hasattr(m, 'reparam'):
            delattr(m, 'reparam')

tp.utils = types.SimpleNamespace(remove_pruning_reparametrization=remove_pruning_reparametrization)
sys.modules['torch_pruning'] = tp

class DummyYOLO:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 3),
            torch.nn.ReLU(),
        )
        self.callbacks = {{}}
    def add_callback(self, event, cb):
        self.callbacks.setdefault(event, []).append(cb)

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda *a, **k: 0
torch_utils.get_num_params = lambda *a, **k: 0
utils.torch_utils = torch_utils
up.utils = utils
up.YOLO = lambda *a, **k: DummyYOLO()
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

from prune_methods.depgraph_hsic import DepgraphHSICMethod
from pipeline.pruning_pipeline_2 import PruningPipeline2
import types as t

method = DepgraphHSICMethod(None, workdir='{tmp_path}')
method.example_inputs = torch.randn(1, 3, 8, 8)
method.logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None)

def analyze_stub(self):
    self.DG = DummyDG()
    self._dg_model = self.model
    self.layers = [self.model[0]]
    self.layer_names = ['0']

method.analyze_model = t.MethodType(analyze_stub, method)
method.pruning_plan = {{'0': [0]}}

pipeline = PruningPipeline2('m', 'd', pruning_method=method)
pipeline.model = DummyYOLO()
method.model = pipeline.model.model

pipeline.apply_pruning()

print(len(calls))
print(hasattr(pipeline.model.model[0], 'reparam'))
"""
    out = subprocess.check_output([sys.executable, '-c', code])
    calls_count, has_reparam = out.decode().strip().splitlines()
    assert calls_count == '1'
    assert has_reparam == 'False'
