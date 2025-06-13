"""Lazy import interface for pruning methods."""

from importlib import import_module

__all__ = [
    "BasePruningMethod",
    "L1NormPruningMethod",
    "RandomPruningMethod",
    "DepGraphPruningMethod",
    "TorchPruningRandomMethod",
    "IsomorphicPruningMethod",
    "HSICLassoPruningMethod",
    "DepgraphHSICMethod",
    "WeightedHybridPruningMethod",
]


def __getattr__(name: str):
    if name == "BasePruningMethod":
        return import_module("prune_methods.base").BasePruningMethod
    if name == "L1NormPruningMethod":
        return import_module("prune_methods.l1_norm").L1NormPruningMethod
    if name == "RandomPruningMethod":
        return import_module("prune_methods.random_pruning").RandomPruningMethod
    if name == "DepGraphPruningMethod":
        return import_module("prune_methods.depgraph_pruning").DepGraphPruningMethod
    if name == "TorchPruningRandomMethod":
        return import_module("prune_methods.torch_pruning_simple").TorchPruningRandomMethod
    if name == "IsomorphicPruningMethod":
        return import_module("prune_methods.isomorphic_pruning").IsomorphicPruningMethod
    if name == "HSICLassoPruningMethod":
        return import_module("prune_methods.hsic_lasso").HSICLassoPruningMethod
    if name == "DepgraphHSICMethod":
        return import_module("prune_methods.depgraph_hsic").DepgraphHSICMethod
    if name == "WeightedHybridPruningMethod":
        return import_module("prune_methods.weighted_hybrid").WeightedHybridPruningMethod
    raise AttributeError(name)
