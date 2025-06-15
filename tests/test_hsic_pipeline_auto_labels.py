import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_generate_mask_after_pipeline_steps(tmp_path):
    code = """
import sys
import types
import torch
from pathlib import Path

mpl = types.ModuleType('matplotlib')
plt = types.ModuleType('matplotlib.pyplot')
sys.modules['matplotlib'] = mpl
sys.modules['matplotlib.pyplot'] = plt

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda m: 0
torch_utils.get_num_params = lambda m: 0
utils.torch_utils = torch_utils
up.utils = utils
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

tp = types.ModuleType('torch_pruning')
class DummyDG:
    def build_dependency(self, model, example_inputs):
        pass
    def get_all_groups(self, root_module_types=None):
        return []
    def get_pruner_of_module(self, layer):
        return types.SimpleNamespace(get_out_channels=lambda l: getattr(l, 'out_channels', 0))
    def get_pruning_group(self, conv, fn, idxs):
        return []
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

    def train(self, data=None, **kw):
        for _ in range(2):
            batch = {{"img": torch.randn(1,3,8,8), "cls": torch.tensor([1.0])}}
            self.model(batch["img"])
            trainer = types.SimpleNamespace(batch=batch)
            for cb in self.callbacks.get("on_train_batch_end", []):
                cb(trainer)
        return {{}}

up.YOLO = lambda *a, **k: DummyYOLO()

from pipeline.context import PipelineContext
from pipeline import (
    LoadModelStep,
    CalcStatsStep,
    AnalyzeModelStep,
    TrainStep,
    GenerateMasksStep,
)
from prune_methods.depgraph_hsic import DepgraphHSICMethod


ctx = PipelineContext('m', 'd', workdir=Path('{tmp}'))
method = DepgraphHSICMethod(None, workdir='{tmp}')
ctx.pruning_method = method

steps = [
    LoadModelStep(),
    CalcStatsStep('initial'),
    AnalyzeModelStep(),
    TrainStep('pretrain', epochs=1, plots=False),
    GenerateMasksStep(ratio=0.5),
]

for i, step in enumerate(steps):
    step.run(ctx)
    if i == 0:
        ctx.pruning_method.model = ctx.model.model
print('ok')
""".format(tmp=tmp_path)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ok" in proc.stdout
