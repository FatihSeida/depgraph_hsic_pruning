import builtins
import os
import sys
import types
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper.experiment_manager import ExperimentManager


def test_compare_pruning_methods_without_matplotlib(monkeypatch, tmp_path):
    mgr = ExperimentManager("yolo", workdir=tmp_path)
    mgr.add_result("dummy", 0.5, {"mAP": 0.1})

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"matplotlib.pyplot", "seaborn"}:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    mgr.compare_pruning_methods()  # Should handle missing matplotlib gracefully


def test_compare_pruning_methods_multiple_lines(monkeypatch, tmp_path):
    mgr = ExperimentManager("yolo", workdir=tmp_path)
    mgr.add_result("m1", 0.1, {"mAP": 0.2})
    mgr.add_result("m1", 0.2, {"mAP": 0.3})
    mgr.add_result("m2", 0.1, {"mAP": 0.25})

    records = []

    dummy = types.SimpleNamespace(
        figure=lambda: None,
        title=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        legend=lambda *a, **k: None,
        tight_layout=lambda: None,
        savefig=lambda *a, **k: None,
        close=lambda *a, **k: None,
    )

    def plot(x, y, marker=None, label=None):
        records.append({"x": x, "y": y, "label": label})

    dummy.plot = plot

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", dummy)

    mgr.compare_pruning_methods()

    assert len(records) == 2
    assert {r["label"] for r in records} == {"m1", "m2"}


def test_plot_functions_without_matplotlib(monkeypatch, tmp_path):
    mgr = ExperimentManager("yolo", workdir=tmp_path)
    mgr.add_result("m1", 0.1, {"training": {"mAP": 0.2}})
    mgr.add_result("m2", 0.2, {"training": {"mAP": 0.3}})

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"matplotlib.pyplot", "seaborn"}:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    mgr.plot_line("training.mAP")
    mgr.plot_heatmap("training.mAP")


def test_extract_metric_numpy_float(tmp_path):
    mgr = ExperimentManager("yolo", workdir=tmp_path)
    data = {"training": {"score": np.float32(0.75)}}
    assert mgr._extract_metric(data, "training.score") == 0.75


def test_heatmap_created_for_nested_metrics(tmp_path):
    # ensure real matplotlib and seaborn are available
    for mod in ["matplotlib.pyplot", "matplotlib", "seaborn"]:
        sys.modules.pop(mod, None)
    import importlib
    importlib.import_module("matplotlib.pyplot")
    importlib.import_module("seaborn")

    mgr = ExperimentManager("yolo", workdir=tmp_path)
    metrics = {
        "training": {"mAP": 0.2},
        "pruning": {"parameters": {"reduction_percent": 50.0}},
    }
    mgr.add_result("m1", 0.5, metrics)

    mgr.plot_heatmap("pruning.parameters.reduction_percent")

    out = tmp_path / "pruning_parameters_reduction_percent_heatmap.png"
    assert out.exists()

    # cleanup to avoid interfering with other tests
    for mod in list(sys.modules):
        if mod.startswith("matplotlib") or mod.startswith("seaborn"):
            sys.modules.pop(mod, None)
