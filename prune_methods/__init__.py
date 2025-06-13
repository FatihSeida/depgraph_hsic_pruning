"""Lazy import interface for pruning methods."""

from importlib import import_module

__all__ = [
    "BasePruningMethod",
    "L1NormMethod",
    "RandomMethod",
    "DepgraphMethod",
    "TorchRandomMethod",
    "IsomorphicMethod",
    "HSICLassoMethod",
    "DepgraphHSICMethod",
    "WeightedHybridMethod",
]


def __getattr__(name: str):
    if name == "BasePruningMethod":
        return import_module("prune_methods.base").BasePruningMethod
    if name == "L1NormMethod":
        return import_module("prune_methods.l1_norm").L1NormMethod
    if name == "RandomMethod":
        return import_module("prune_methods.random_pruning").RandomMethod
    if name == "DepgraphMethod":
        return import_module("prune_methods.depgraph_pruning").DepgraphMethod
    if name == "TorchRandomMethod":
        return import_module("prune_methods.torch_pruning_simple").TorchRandomMethod
    if name == "IsomorphicMethod":
        return import_module("prune_methods.isomorphic_pruning").IsomorphicMethod
    if name == "HSICLassoMethod":
        return import_module("prune_methods.hsic_lasso").HSICLassoMethod
    if name == "DepgraphHSICMethod":
        return import_module("prune_methods.depgraph_hsic").DepgraphHSICMethod
    if name == "WeightedHybridMethod":
        return import_module("prune_methods.weighted_hybrid").WeightedHybridMethod
    raise AttributeError(name)
