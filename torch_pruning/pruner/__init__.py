"""Stub pruner implementations for the unit tests.

They only mimic the interfaces of the real :mod:`torch_pruning` pruners.
Install ``torch-pruning`` for production-grade pruning algorithms.
"""

from .algorithms import BasePruner as BasePruner

class RandomPruner(BasePruner):
    pass

class MagnitudePruner(BasePruner):
    pass
