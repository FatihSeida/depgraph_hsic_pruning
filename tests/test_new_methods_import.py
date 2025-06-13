import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# stub heavy dependency
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("torch_pruning", types.ModuleType("torch_pruning"))
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
linear_model_stub = types.ModuleType("sklearn.linear_model")
class LassoLars:  # pragma: no cover - placeholder
    pass
linear_model_stub.LassoLars = LassoLars
sys.modules.setdefault("sklearn.linear_model", linear_model_stub)


def test_new_methods_have_flag():
    names = [
        "L1NormMethod",
        "RandomMethod",
        "DepgraphMethod",
        "TorchRandomMethod",
        "IsomorphicMethod",
        "HSICLassoMethod",
        "DepgraphHSICMethod",
        "WeightedHybridMethod",
    ]
    mod = __import__("prune_methods", fromlist=names)
    for name in names:
        assert hasattr(mod, name)
    DepGraph = mod.DepgraphMethod
    Simple = mod.TorchRandomMethod
    Iso = mod.IsomorphicMethod
    HSIC = mod.DepgraphHSICMethod
    assert DepGraph.requires_reconfiguration is False
    assert Simple.requires_reconfiguration is False
    assert Iso.requires_reconfiguration is False
    assert HSIC.requires_reconfiguration is False

