import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_pruning_after_layer_replacement(tmp_path):
    code = f"""
import torch
from prune_methods.depgraph_hsic import DepgraphHSICMethod
import torch_pruning as tp

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 8, 3),
    torch.nn.ReLU(),
)
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = torch.randn(1, 3, 8, 8)
method.analyze_model()
model[0] = torch.nn.Conv2d(3, 4, 3)
conv = model[0]
method.pruning_plan = [method.DG.get_pruning_group(conv, tp.prune_conv_out_channels, [0])]
method.apply_pruning()
print('ok')
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert 'ok' in proc.stdout
