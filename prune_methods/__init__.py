"""Pruning method interfaces and placeholders."""

from .base import BasePruningMethod
from .l1_norm import L1NormPruningMethod
from .random_pruning import RandomPruningMethod
__all__ = [
    "BasePruningMethod",
    "L1NormPruningMethod",
    "RandomPruningMethod",
]
