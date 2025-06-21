import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_warning_for_few_labels(tmp_path):
    code = f"""
import logging
import torch
from prune_methods.depgraph_hsic import DepgraphHSICMethod

logging.basicConfig(level=logging.WARNING)

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 8, 3),
    torch.nn.ReLU(),
)
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = torch.randn(1, 3, 8, 8)
method.analyze_model()
model(torch.randn(1, 3, 8, 8))
method.add_labels(torch.tensor([1.0]))
method.generate_pruning_mask(0.5)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    output = proc.stderr + proc.stdout
    assert "HSIC computation may be invalid" in output
    assert "L1-norm" in output

