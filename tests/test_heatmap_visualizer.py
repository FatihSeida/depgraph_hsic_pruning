import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper.heatmap_visualizer import plot_metric_heatmaps
import importlib


def test_heatmap_images_created(tmp_path):
    # Ensure real matplotlib and seaborn are available in case other tests
    # stubbed them out.
    for mod in ["matplotlib.pyplot", "matplotlib", "seaborn"]:
        sys.modules.pop(mod, None)
    importlib.import_module("matplotlib.pyplot")
    importlib.import_module("seaborn")

    df = pd.DataFrame({
        "pruning.method": ["m1", "m1", "m2", "m2"],
        "pruning.ratio": [0.1, 0.2, 0.1, 0.2],
        "FLOPsReduction": [0.1, 0.2, 0.3, 0.4],
        "FilterReduction": [1, 2, 3, 4],
    })

    metrics = ["FLOPsReduction", "FilterReduction"]
    plot_metric_heatmaps(df, metrics, tmp_path)

    for metric in metrics:
        safe_metric = metric.replace(".", "_")
        name = f"{safe_metric}_heatmap.png"
        assert (tmp_path / name).exists()
