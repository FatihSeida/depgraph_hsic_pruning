import subprocess
import sys
from pathlib import Path


def test_generate_mask_no_baseline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = f"""
import sys
import types
import torch
from pathlib import Path

sys.path.insert(0, '{root.as_posix()}')

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda *a, **k: 0
torch_utils.get_num_params = lambda *a, **k: 0
utils.torch_utils = torch_utils
up.YOLO = lambda *a, **k: None
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 3),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)

    def parameters(self):  # pragma: no cover - torch defines this
        return super().parameters()

    def train(self, *a, **kw):
        return {{}}


class DummyYOLO:
    def __init__(self, *a, **k):
        self.model = DummyModel()
        self.callbacks = {{}}

    def add_callback(self, event, cb):
        self.callbacks.setdefault(event, []).append(cb)

    def train(self, *a, **kw):
        return {{}}


up.utils = utils
up.YOLO = lambda *a, **k: DummyYOLO()

import main
from prune_methods.depgraph_hsic import DepgraphHSICMethod

yaml_mod = types.ModuleType('yaml')
yaml_mod.safe_load = lambda f: {{"path": str(Path(f.name).parent), "val": "images"}}
sys.modules['yaml'] = yaml_mod

tmp = Path("{tmp_path}")
(tmp / 'images').mkdir()
(tmp / 'images' / 'img.jpg').write_text('x')
(tmp / 'labels').mkdir()
(tmp / 'labels' / 'img.txt').write_text('0')
data_file = tmp / 'd.yaml'
data_file.write_text('path: .\\nval: images')

cfg = main.TrainConfig(baseline_epochs=0, finetune_epochs=0, batch_size=1, ratios=[0.2])
main.execute_pipeline('m.pt', str(data_file), DepgraphHSICMethod, 0.2, cfg, tmp)
print('ok')
"""

    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert 'ok' in proc.stdout

