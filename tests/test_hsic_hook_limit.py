import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_hooks_only_first_modules(tmp_path):
    code = f"""
import json
import sys
import torch
from prune_methods.depgraph_hsic import DepgraphHSICMethod

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.Conv2d(3, 3, 1) for _ in range(12)])

model = Dummy()
method = DepgraphHSICMethod(model, workdir='{tmp_path}')
method.register_hooks()
json.dump({{"count": len(method.layers), "names": method.layer_names}}, sys.stdout)
"""
    out = subprocess.check_output([sys.executable, "-c", code])
    data = json.loads(out.decode())
    assert data['count'] == 10
    assert data['names'][0] == "model.0"
    assert data['names'][-1] == "model.9"
