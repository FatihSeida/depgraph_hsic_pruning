import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_final_retry_after_analyze(tmp_path):
    code = f"""
import json
import sys
import types
import torch

sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

class DummyGroup(list):
    def __init__(self, conv, idxs):
        super().__init__([(types.SimpleNamespace(target=types.SimpleNamespace(module=conv)), idxs)])
        self.conv = conv
        self.idxs = idxs
    def prune(self):
        mask = torch.ones(self.conv.out_channels, dtype=torch.bool)
        for i in self.idxs:
            mask[i] = False
        self.conv.out_channels = int(mask.sum())
        self.conv.weight = torch.nn.Parameter(self.conv.weight[mask])
        if self.conv.bias is not None:
            self.conv.bias = torch.nn.Parameter(self.conv.bias[mask])

class DummyDG:
    calls = 0
    def build_dependency(self, model, example_inputs):
        pass
    def get_all_groups(self, root_module_types=None):
        return []
    def get_pruner_of_module(self, layer):
        return types.SimpleNamespace(get_out_channels=lambda l: getattr(l, 'out_channels', 0))
    def get_pruning_group(self, conv, fn, idxs):
        DummyDG.calls += 1
        if DummyDG.calls < 3:
            raise ValueError('fail')
        return DummyGroup(conv, idxs)

tp = types.ModuleType('torch_pruning')
tp.DependencyGraph = DummyDG
tp.prune_conv_out_channels = lambda *a, **k: None
tp.utils = types.SimpleNamespace(remove_pruning_reparametrization=lambda m: None)
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
method.analyze_model()
DummyDG.calls = 0
start = DummyDG.calls
model[0] = torch.nn.Conv2d(3,4,3)
conv = model[0]
method.pruning_plan = [method.DG.get_pruning_group(conv, None, [0])]
method.apply_pruning()
print(json.dumps([DummyDG.calls - start]))
"""
    out = subprocess.check_output([sys.executable, '-c', code])
    calls, = json.loads(out.decode())
    assert calls == 3
