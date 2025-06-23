import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_error_for_insufficient_labels(tmp_path):
    code = f"""
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
for _ in range(3):
    model(torch.randn(1, 3, 8, 8))
    method.add_labels(torch.tensor([1.0]))
method.generate_pruning_mask(0.5)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Insufficient labels" in proc.stderr + proc.stdout

