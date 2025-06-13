import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run_method(module: str, cls: str):
    code = f"""
import json
import torch
import sys
from prune_methods.{module} import {cls}
model = torch.nn.Sequential(
    torch.nn.Conv2d(3,4,3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4,8,3),
    torch.nn.ReLU(),
)
method = {cls}(model)
method.example_inputs = torch.randn(1,3,8,8)
method.analyze_model()
before=sum(p.numel() for p in model.parameters())
method.generate_pruning_mask(0.5)
method.apply_pruning()
after=sum(p.numel() for p in model.parameters())
json.dump([before, after], sys.stdout)
"""
    out = subprocess.check_output([sys.executable, "-c", code])
    return json.loads(out.decode())


def test_depgraph_method_reduces_params():
    before, after = _run_method("depgraph_pruning", "DepgraphMethod")
    assert after < before


def test_random_method_reduces_params():
    before, after = _run_method("torch_pruning_simple", "TorchRandomMethod")
    assert after < before
