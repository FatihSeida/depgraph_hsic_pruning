"""Simplified placeholder for tests.

This stub mimics a tiny portion of the real :mod:`torch_pruning` package so
the unit tests can run without the actual dependency. Install the real
``torch-pruning`` library for any production usage.
"""

import types
import importlib.metadata as _metadata
import torch
from torch import nn
from types import SimpleNamespace
from . import utils
from . import importance
from . import pruner

__version__ = '1.5.0'

# expose modules
DependencyGraph = None
prune_conv_out_channels = None

class DependencyGraph:
    def build_dependency(self, model, example_inputs):
        pass

    def get_all_groups(self, root_module_types=None):
        return []

    def get_pruner_of_module(self, layer):
        return SimpleNamespace(get_out_channels=lambda l: getattr(l, 'out_channels', 0))

    def get_pruning_group(self, conv, fn, idxs):
        class Group(list):
            def __init__(self):
                super().__init__([(SimpleNamespace(target=SimpleNamespace(module=conv)), idxs)])
            def prune(self):
                mask = torch.ones(conv.out_channels, dtype=torch.bool)
                for i in sorted(idxs):
                    if i < conv.out_channels:
                        mask[i] = False
                conv.out_channels = int(mask.sum())
                conv.weight = nn.Parameter(conv.weight[mask])
                if conv.bias is not None:
                    conv.bias = nn.Parameter(conv.bias[mask])
        return Group()

DependencyGraph = DependencyGraph

def prune_conv_out_channels(module, idxs, dim=0):
    pass

# patch importlib.metadata.version to return __version__
_real_version = _metadata.version

def _version(pkg):
    if pkg == 'torch_pruning':
        return __version__
    return _real_version(pkg)

_metadata.version = _version
