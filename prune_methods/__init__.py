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
    "DepGraphHSICMethod2",
    "WeightedHybridMethod",
]

_MAPPING = {
    "BasePruningMethod": ("prune_methods.base", "BasePruningMethod"),
    "L1NormMethod": ("prune_methods.l1_norm", "L1NormMethod"),
    "RandomMethod": ("prune_methods.random_pruning", "RandomMethod"),
    "DepgraphMethod": ("prune_methods.depgraph_pruning", "DepgraphMethod"),
    "TorchRandomMethod": ("prune_methods.torch_pruning_simple", "TorchRandomMethod"),
    "IsomorphicMethod": ("prune_methods.isomorphic_pruning", "IsomorphicMethod"),
    "HSICLassoMethod": ("prune_methods.hsic_lasso", "HSICLassoMethod"),
    "DepgraphHSICMethod": ("prune_methods.depgraph_hsic", "DepgraphHSICMethod"),
    "DepGraphHSICMethod2": ("prune_methods.depgraph_hsic_2", "DepGraphHSICMethod2"),
    "WeightedHybridMethod": ("prune_methods.weighted_hybrid", "WeightedHybridMethod"),
}


def __getattr__(name: str):
    if name in _MAPPING:
        module, attr = _MAPPING[name]
        obj = getattr(import_module(module), attr)
        if name.endswith("Method") and not hasattr(obj, "requires_reconfiguration"):
            obj.requires_reconfiguration = False
        return obj
    raise AttributeError(name)
