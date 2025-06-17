import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_apply_after_layer_swap(tmp_path):
    code = f"""
import json
import importlib.metadata
import sys
import torch
import torch_pruning as tp
from prune_methods.depgraph_hsic import DepgraphHSICMethod

ver = tuple(map(int, importlib.metadata.version('torch_pruning').split('.')[:2]))
assert ver >= (1, 5)

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 8, 3),
    torch.nn.ReLU(),
)
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = torch.randn(1, 3, 8, 8)
method.analyze_model()
for _ in range(2):
    model(torch.randn(1, 3, 8, 8))
    method.add_labels(torch.tensor([1.0]))
method.generate_pruning_mask(0.5)
model[0] = torch.nn.Conv2d(3, 4, 3)
before = sum(p.numel() for p in model.parameters())
method.apply_pruning()
after = sum(p.numel() for p in model.parameters())
json.dump([before, after], sys.stdout)
"""
    out = subprocess.check_output([sys.executable, '-c', code])
    before, after = json.loads(out.decode())
    assert after < before
