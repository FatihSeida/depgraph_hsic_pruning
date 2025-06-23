import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_pipeline_prunes_after_inplace_change(tmp_path):
    code = """
import json
import sys
import types
import torch

sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

class DummyDG:
    def build_dependency(self, model, example_inputs):
        self.model = model

    def get_all_groups(self, root_module_types=None):
        groups = []
        for layer in getattr(self, "model", []).modules():
            if isinstance(layer, torch.nn.Conv2d):
                groups.append(self.get_pruning_group(layer, None, [0]))
        return groups
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

class DummyYOLO:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3,4,3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4,8,3),
            torch.nn.ReLU(),
        )
        self.callbacks = {{}}
    def add_callback(self, event, cb):
        self.callbacks.setdefault(event, []).append(cb)
    def train(self, *a, **k):
        self.model[0] = torch.nn.Conv2d(3,4,3)
        return {{}}

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_num_params = lambda *a, **k: 0
utils.torch_utils = torch_utils
from helper import flops_utils as fu
fu.get_flops_reliable = lambda *a, **k: 0
up.utils = utils
up.YOLO = lambda *a, **k: DummyYOLO()
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

from prune_methods.depgraph_hsic import DepgraphHSICMethod
from pipeline.pruning_pipeline import PruningPipeline

method = DepgraphHSICMethod(None, workdir='{tmp}')
pipeline = PruningPipeline('m', 'd', pruning_method=method)
pipeline.model = DummyYOLO()
method.model = pipeline.model.model
method.example_inputs = torch.randn(1,3,8,8)
pipeline.analyze_structure()
method.remove_hooks()
pipeline.pretrain()
pipeline.analyze_structure()
for _ in range(2):
    pipeline.model.model(torch.randn(1,3,8,8))
    method.add_labels(torch.tensor([1.0]))
pipeline.generate_pruning_mask(0.5)
before = sum(p.numel() for p in pipeline.model.model.parameters())
pipeline.apply_pruning()
after = sum(p.numel() for p in pipeline.model.model.parameters())
json.dump([before, after], sys.stdout)
""".format(tmp=tmp_path)
    out = subprocess.check_output([sys.executable, "-c", code])
    before, after = json.loads(out.decode())
    assert after < before
