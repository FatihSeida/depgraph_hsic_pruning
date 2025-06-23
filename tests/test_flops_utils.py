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


def test_calculate_flops_manual_respects_device(monkeypatch):
    import torch
    import torch.nn as nn

    fu = importlib.import_module('helper.flops_utils')
    importlib.reload(fu)

    model = nn.Sequential(nn.Conv2d(3, 8, 1))
    device = next(model.parameters()).device

    recorded = {}
    orig_zeros = torch.zeros

    def spy_zeros(*args, **kwargs):
        recorded["device"] = kwargs.get("device")
        return orig_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", spy_zeros)

    fu.calculate_flops_manual(model, imgsz=8)

    assert recorded["device"] == device


def test_calculate_flops_manual_handles_errors(monkeypatch):
    import torch
    import torch.nn as nn

    fu = importlib.import_module('helper.flops_utils')
    importlib.reload(fu)

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 13, 1)
            self.conv2 = nn.Conv2d(16, 8, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)  # channel mismatch triggers error
            return x

    model = BadModel()

    flops = fu.calculate_flops_manual(model, imgsz=8)

    assert flops == 0.0
