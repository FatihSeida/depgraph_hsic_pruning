import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_layerwise_pruning_forward(tmp_path):
    code = """
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from prune_methods.depgraph_hsic_2 import DepGraphHSICMethod2

class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        return self.bn2(self.conv2(y)) + y

model = TinyResidual()
method = DepGraphHSICMethod2(model, workdir='{tmp}')
method.example_inputs = torch.randn(1, 3, 8, 8)
method.analyze_model()
dataloader = DataLoader(TensorDataset(torch.randn(1, 3, 8, 8), torch.tensor([1.0])), batch_size=1)
method.generate_pruning_mask(0.5, dataloader)
res = {{'fallback': method.fallback_layerwise}}
method.apply_pruning()
out = model(torch.randn(1, 3, 8, 8))
res['out_shape'] = list(out.shape)
print(json.dumps(res))
""".format(tmp=str(tmp_path))
    out = subprocess.check_output([sys.executable, "-c", code])
    data = json.loads(out.decode())
    assert data['fallback'] is True
    assert data['out_shape'][1] > 0
