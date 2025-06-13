import os
import sys
import pandas as pd
import importlib
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper import load_metrics_dataframe
from helper.heatmap_visualizer import plot_default_metric_heatmaps, DEFAULT_METRICS


def test_load_metrics_dataframe(tmp_path):
    run1 = tmp_path / "MethodA_r0.1"
    run1.mkdir()
    (run1 / "metrics.csv").write_text("a,b\n1,2\n")

    run2 = tmp_path / "MethodB_r0_2"
    run2.mkdir()
    (run2 / "metrics.csv").write_text("a,b\n3,4\n")

    df = load_metrics_dataframe(tmp_path)
    assert set(df["pruning.method"]) == {"MethodA", "MethodB"}
    assert set(df["pruning.ratio"]) == {0.1, 0.2}
    assert set(df.columns) >= {"a", "b", "pruning.method", "pruning.ratio"}


def test_plot_default_metric_heatmaps(tmp_path, monkeypatch):
    calls = []

    def fake(df, metrics, output_dir):
        calls.append((list(metrics), Path(output_dir)))

    monkeypatch.setattr(
        "helper.heatmap_visualizer.plot_metric_heatmaps", fake, raising=False
    )

    df = pd.DataFrame({
        "pruning.method": ["m1", "m1", "m2", "m2"],
        "pruning.ratio": [0.1, 0.2, 0.1, 0.2],
        DEFAULT_METRICS[0]: [1, 2, 3, 4],
        DEFAULT_METRICS[1]: [5, 6, 7, 8],
        DEFAULT_METRICS[2]: [9, 10, 11, 12],
    })

    plot_default_metric_heatmaps(df, tmp_path)

    assert calls == [(DEFAULT_METRICS, tmp_path)]
