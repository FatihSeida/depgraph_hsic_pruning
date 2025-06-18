import subprocess, sys, types
from pathlib import Path

def test_remove_pruning_reparam_called(tmp_path):
    code = f"""
import torch
import types
import sys

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

from prune_methods.depgraph_hsic import DepgraphHSICMethod
model = torch.nn.Sequential(
    torch.nn.Conv2d(3,4,3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4,8,3),
    torch.nn.ReLU(),
)
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = torch.randn(1,3,8,8)
method.DG = DummyDG()
method.pruning_plan = {{'0': [0]}}
method.apply_pruning()
print(len(calls))
print(hasattr(model[0], 'reparam'))
"""
    out = subprocess.check_output([sys.executable, '-c', code])
    calls_count, has_reparam = out.decode().strip().splitlines()
    assert calls_count == '1'
    assert has_reparam == 'False'
