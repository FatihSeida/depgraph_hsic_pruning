import importlib
import types
import sys


def test_calculate_flops_manual_no_torch(monkeypatch):
    torch_stub = types.ModuleType('torch')
    torch_nn_stub = types.ModuleType('torch.nn')
    monkeypatch.setitem(sys.modules, 'torch', torch_stub)
    monkeypatch.setitem(sys.modules, 'torch.nn', torch_nn_stub)
    fu = importlib.import_module('helper.flops_utils')
    importlib.reload(fu)
    assert fu.calculate_flops_manual(None) == 0.0


def test_get_flops_reliable_fallback(monkeypatch):
    import torch.nn as nn
    model = nn.Sequential(nn.Conv2d(3, 8, 1))
    fu = importlib.import_module("helper.flops_utils")
    importlib.reload(fu)
    monkeypatch.setattr(fu, "_get_flops", lambda *a, **k: 0, raising=False)
    flops = fu.get_flops_reliable(model, imgsz=8)
    assert flops > 0


def test_get_num_params_reliable_fallback(monkeypatch):
    model = types.SimpleNamespace(
        parameters=lambda: [
            types.SimpleNamespace(numel=lambda: 4),
            types.SimpleNamespace(numel=lambda: 6),
        ]
    )
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    up.utils = utils
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    fu = importlib.import_module('helper.flops_utils')
    importlib.reload(fu)
    params = fu.get_num_params_reliable(model)
    assert params == 10
