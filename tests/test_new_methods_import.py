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


def test_new_methods_have_flag():
    mod = __import__("prune_methods", fromlist=["DepGraphPruningMethod", "TorchPruningRandomMethod"])
    DepGraph = mod.DepGraphPruningMethod
    Simple = mod.TorchPruningRandomMethod
    assert DepGraph.requires_reconfiguration is False
    assert Simple.requires_reconfiguration is False

