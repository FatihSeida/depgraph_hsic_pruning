import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_warning_for_missing_layers(tmp_path):
    code = f"""
import logging
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
        if getattr(layer, 'skip', False):
            return None
        return types.SimpleNamespace(get_out_channels=lambda l: getattr(l, 'out_channels', 0))
    def get_pruning_group(self, conv, fn, idxs):
        class Group(list):
            def prune(self):
                pass
        return Group()

sys.modules['torch_pruning'] = types.ModuleType('torch_pruning')
sys.modules['torch_pruning'].DependencyGraph = DummyDG
sys.modules['torch_pruning'].prune_conv_out_channels = lambda *a, **k: None
sys.modules['torch_pruning'].utils = types.SimpleNamespace(remove_pruning_reparametrization=lambda m: None)

from prune_methods.depgraph_hsic import DepgraphHSICMethod

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 8, 3),
    torch.nn.ReLU(),
)
model[0].skip = True
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = torch.randn(1, 3, 8, 8)
logging.basicConfig(level=logging.WARNING)
method.analyze_model()
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    output = proc.stderr + proc.stdout
    assert "Dependency graph missing layers" in output
    assert "'0'" in output
