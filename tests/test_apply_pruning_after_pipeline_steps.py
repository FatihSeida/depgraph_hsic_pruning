import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_apply_pruning_after_pipeline_steps(tmp_path):
    code = f"""
import sys
import types
import torch
from pathlib import Path

# minimal stubs to avoid heavy deps
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

# mock torch_pruning
class DummyGroup:
    def prune(self):
        pass

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
                pass
        return Group()
    def prune_group(self, group):
        group.prune()

tp = types.ModuleType('torch_pruning')
tp.DependencyGraph = DummyDG
tp.prune_conv_out_channels = lambda *a, **k: None
tp.utils = types.SimpleNamespace(remove_pruning_reparametrization=lambda m: None)
sys.modules['torch_pruning'] = tp

# dummy ultralytics YOLO
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
    def train(self, data=None, **kw):
        for _ in range(2):
            batch = {{"img": torch.randn(1,3,8,8), "cls": torch.tensor([1.0])}}
            self.model(batch["img"])
            trainer = types.SimpleNamespace(batch=batch)
            for cb in self.callbacks.get("on_train_batch_end", []):
                cb(trainer)
        return {{}}

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda m: 0
torch_utils.get_num_params = lambda m: 0
utils.torch_utils = torch_utils
up.utils = utils
up.YOLO = lambda *a, **k: DummyYOLO()
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

from pipeline.context import PipelineContext
from pipeline import (
    LoadModelStep,
    AnalyzeModelStep,
    TrainStep,
    GenerateMasksStep,
    ApplyPruningStep,
)
from prune_methods.depgraph_hsic import DepgraphHSICMethod

ctx = PipelineContext('m', 'd', workdir=Path('{tmp_path}'))
method = DepgraphHSICMethod(None, workdir='{tmp_path}')
method.example_inputs = torch.randn(1,3,8,8)
def analyze_stub(self):
    self.DG = DummyDG()
    self._dg_model = self.model
    self.layers = [self.model[0]]
    self.layer_names = ['0']
method.analyze_model = types.MethodType(analyze_stub, method)
def mask_stub(self, ratio):
    conv = self.layers[0]
    self.pruning_plan = [self.DG.get_pruning_group(conv, None, [0])]
method.generate_pruning_mask = types.MethodType(mask_stub, method)
ctx.pruning_method = method

steps = [
    LoadModelStep(),
    AnalyzeModelStep(),
    TrainStep('pretrain', epochs=1, plots=False),
    GenerateMasksStep(ratio=0.5),
    ApplyPruningStep(),
]

for step in steps:
    step.run(ctx)

print('ok')
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ok" in proc.stdout
