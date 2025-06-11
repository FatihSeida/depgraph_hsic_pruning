"""Pruning method interfaces and placeholders."""

from .base import BasePruningMethod
from .l1_norm import L1NormPruningMethod
from .random_pruning import RandomPruningMethod
from .methods import (
    Method3,
    Method4,
    Method5,
    Method6,
    Method7,
    Method8,
)

__all__ = [
    "BasePruningMethod",
    "L1NormPruningMethod",
    "RandomPruningMethod",
    "Method3",
    "Method4",
    "Method5",
    "Method6",
    "Method7",
    "Method8",
]
