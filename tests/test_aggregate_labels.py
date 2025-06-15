import os
import sys
import types
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _import_main(monkeypatch):
    if 'torch' in sys.modules and not hasattr(sys.modules['torch'], '__file__'):
        sys.modules.pop('torch')
    if 'torch.nn' in sys.modules and not hasattr(sys.modules['torch.nn'], '__file__'):
        sys.modules.pop('torch.nn')
    import torch
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", torch.nn)

    up = types.ModuleType("ultralytics")
    utils = types.ModuleType("ultralytics.utils")
    torch_utils = types.ModuleType("ultralytics.utils.torch_utils")
    torch_utils.get_flops = lambda *a, **k: 0
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    up.utils = utils
    up.YOLO = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ultralytics", up)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", torch_utils)

    return importlib.import_module("main")


def test_aggregate_labels_returns_one_per_image(monkeypatch):
    main = _import_main(monkeypatch)
    import torch
    batch = {"img": torch.zeros(2, 3, 1, 1), "cls": torch.tensor([1, 2])}
    out = main.aggregate_labels(batch)
    assert out.shape[0] == 2
    assert torch.equal(out, torch.tensor([1, 2]))


def test_aggregate_labels_handles_multi_object_batches(monkeypatch):
    main = _import_main(monkeypatch)
    import torch
    batch = {
        "img": torch.zeros(2, 3, 1, 1),
        "cls": torch.tensor([1, 1, 2, 3]),
        "batch_idx": torch.tensor([0, 0, 1, 1]),
    }
    out = main.aggregate_labels(batch)
    assert out.shape[0] == 2
    assert out.tolist() == [1, 2]
