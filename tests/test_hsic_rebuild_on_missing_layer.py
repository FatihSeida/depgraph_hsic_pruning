import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_hsic_rebuild_on_missing_layer(tmp_path):
    code = f"""
import json
import sys
import types
import torch

sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

class DummyDG:
    def build_dependency(self, model, example_inputs):
        pass
    def get_all_groups(self, root_module_types=None):
        return []
    def get_pruner_of_module(self, layer):
        return types.SimpleNamespace(get_out_channels=lambda l: getattr(l, 'out_channels', 0))
    def get_pruning_group(self, conv, fn, idxs):
        class Group(list):
            def __init__(self):
                super().__init__([(types.SimpleNamespace(target=types.SimpleNamespace(module=conv)), idxs)])
            def prune(self):
                mask = torch.ones(conv.out_channels, dtype=torch.bool)
                for i in idxs:
                    mask[i] = False
                conv.out_channels = int(mask.sum())
                conv.weight = torch.nn.Parameter(conv.weight[mask])
                if conv.bias is not None:
                    conv.bias = torch.nn.Parameter(conv.bias[mask])
        return Group()

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
model[0] = torch.nn.Conv2d(3,4,3)
method.remove_hooks()
method.model = model
method.analyze_model()
for _ in range(2):
    model(torch.randn(1,3,8,8))
    method.add_labels(torch.tensor([1.0]))
method.generate_pruning_mask(0.5)
before = sum(p.numel() for p in model.parameters())
method.apply_pruning()
after = sum(p.numel() for p in model.parameters())
json.dump([before, after], sys.stdout)
"""
    out = subprocess.check_output([sys.executable, "-c", code])
    before, after = json.loads(out.decode())
    assert after < before
