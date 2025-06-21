import json
import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_compute_channel_wise_hsic_values():
    code = """
import json
import sys
import torch
from prune_methods.hsic_lasso import compute_channel_wise_hsic
feats = torch.tensor([
    [[1.0, 0.0]],
    [[1.0, 1.0]],
    [[1.0, 2.0]],
    [[1.0, 3.0]],
]).view(4, 2, 1, 1)
labels = torch.tensor([0.0, 1.0, 2.0, 3.0])
vals = compute_channel_wise_hsic(feats, labels, gamma=1.0)
json.dump(vals.tolist(), sys.stdout)
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    vals = json.loads(out)
    assert pytest.approx(vals[0], abs=1e-6) == 0.0
    assert pytest.approx(vals[1], rel=1e-3) == 0.1425


def test_generate_pruning_mask_keeps_highest_score(tmp_path):
    code = f"""
import json
import sys
import torch
from prune_methods.hsic_lasso import HSICLassoMethod
model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1))
method = HSICLassoMethod(model, workdir='{tmp_path}')
method.layers = [(model, '0', None)]
feats = torch.tensor([
    [[1.0, 0.0]],
    [[1.0, 1.0]],
    [[1.0, 2.0]],
    [[1.0, 3.0]],
]).view(4, 2, 1, 1)
labels = torch.tensor([0.0, 1.0, 2.0, 3.0])
method.set_layer_data(0, feats, labels)
method.generate_pruning_mask(0.5)
json.dump(method.masks[0].tolist(), sys.stdout)
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    mask = json.loads(out)
    assert mask == [False, True]
