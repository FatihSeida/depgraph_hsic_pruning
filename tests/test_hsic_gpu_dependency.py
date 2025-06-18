import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_hsic_gpu_dependency(tmp_path):
    code = f"""
import json
import torch
import sys
from prune_methods.depgraph_hsic import DepgraphHSICMethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 8, 3),
    torch.nn.ReLU(),
).to(device)
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = torch.randn(1, 3, 8, 8)
method.analyze_model()
json.dump({{'layers': len(method.layers), 'device': str(next(model.parameters()).device), 'expected': str(device)}}, sys.stdout)
"""
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    data = json.loads(proc.stdout)
    assert data['layers'] > 0
    assert data['device'] == data['expected']
