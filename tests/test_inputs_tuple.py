import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run(example, tmp_path):
    code = f"""
import json
import sys
import torch
from prune_methods.depgraph_hsic import DepgraphHSICMethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 8, 3),
    torch.nn.ReLU(),
).to(device)
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.example_inputs = {example}
method.analyze_model()
info = {{'layers': len(method.layers), 'devices': [str(t.device) for t in method._inputs_tuple()], 'expected': str(device)}}
json.dump(info, sys.stdout)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return json.loads(proc.stdout)


def test_dependency_graph_builds_with_list_inputs(tmp_path):
    data = _run('[torch.randn(1,3,8,8)]', tmp_path)
    assert data['layers'] > 0
    assert all(d == data['expected'] for d in data['devices'])


def test_dependency_graph_builds_with_tuple_inputs(tmp_path):
    data = _run('(torch.randn(1,3,8,8),)', tmp_path)
    assert data['layers'] > 0
    assert all(d == data['expected'] for d in data['devices'])
