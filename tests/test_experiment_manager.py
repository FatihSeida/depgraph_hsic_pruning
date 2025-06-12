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
