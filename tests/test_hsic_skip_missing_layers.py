import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_apply_pruning_skips_layers_not_in_dg(tmp_path):
    code = """
import sys
import types
import torch

mpl = types.ModuleType('matplotlib')
plt = types.ModuleType('matplotlib.pyplot')
sys.modules['matplotlib'] = mpl
sys.modules['matplotlib.pyplot'] = plt

# stub torch_pruning
tp = types.ModuleType('torch_pruning')
class DummyDG:
    def __init__(self):
        self.layers = set()
    def build_dependency(self, model, example_inputs):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                self.layers.add(m)
    def get_all_groups(self, root_module_types=None):
        return []
    def get_pruner_of_module(self, layer):
        return types.SimpleNamespace(get_out_channels=lambda l: l.out_channels)
    def get_pruning_group(self, conv, fn, idxs):
        if conv not in self.layers:
            raise ValueError('missing')
        return []

tp.DependencyGraph = DummyDG
tp.prune_conv_out_channels = lambda *a, **k: None
tp.utils = types.SimpleNamespace(remove_pruning_reparametrization=lambda m: None)
sys.modules['torch_pruning'] = tp

# stub sklearn
sk = types.ModuleType('sklearn')
lin = types.ModuleType('sklearn.linear_model')
class DummyLasso:
    def __init__(self, *a, **k):
        self.coef_ = [0]
    def fit(self, X, y):
        pass
lin.LassoLars = DummyLasso
sk.linear_model = lin
sys.modules['sklearn'] = sk
sys.modules['sklearn.linear_model'] = lin

from prune_methods.depgraph_hsic import DepgraphHSICMethod
model = torch.nn.Sequential(torch.nn.Conv2d(3,4,3), torch.nn.ReLU())
method = DepgraphHSICMethod(model, workdir='{tmp}')
method.example_inputs = torch.randn(1,3,8,8)
method.analyze_model()

extra = torch.nn.Conv2d(3,4,3)
method.pruning_plan = {{extra: [0]}}
method.apply_pruning()
print('ok')
""".format(tmp=tmp_path)
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert 'ok' in proc.stdout
