import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_generate_pruning_mask_minimum(tmp_path):
    code = f"""
import json
import sys
import torch
from prune_methods.depgraph_hsic import DepgraphHSICMethod

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
method.generate_pruning_mask(0.05)
json.dump(len(method.pruning_plan), sys.stdout)
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    count = json.loads(out)
    assert count == 1
