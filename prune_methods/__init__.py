"""Pruning method interfaces and placeholders."""

from .base import BasePruningMethod
from .l1_norm import L1NormPruningMethod
from .random_pruning import RandomPruningMethod
from .depgraph_pruning import DepGraphPruningMethod
from .torch_pruning_simple import TorchPruningRandomMethod
__all__ = [
    "BasePruningMethod",
    "L1NormPruningMethod",
    "RandomPruningMethod",
    "DepGraphPruningMethod",
    "TorchPruningRandomMethod",
]
