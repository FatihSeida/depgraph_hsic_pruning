"""Algorithmic stubs for pruning used during testing.

Only minimal logic is provided for unit tests. For full functionality install
the ``torch-pruning`` package.
"""

import torch
from torch import nn

class BasePruner:
    def __init__(self, model, example_inputs=None, importance=None, pruning_ratio=0.5, iterative_steps=1, global_pruning=False, round_to=None):
        self.model = model
        self.pruning_ratio = pruning_ratio

    def step(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                total = m.out_channels
                remove = int(total * self.pruning_ratio)
                if remove <= 0:
                    continue
                keep = torch.ones(total, dtype=torch.bool)
                keep[:remove] = False
                m.out_channels = int(keep.sum())
                m.weight = nn.Parameter(m.weight[keep])
                if m.bias is not None:
                    m.bias = nn.Parameter(m.bias[keep])
