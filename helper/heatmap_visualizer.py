"""Heatmap visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def plot_metric_heatmaps(df: pd.DataFrame, metrics: List[str], output_dir: str) -> None:
    """Generate heatmaps for each metric and save as images.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``pruning.method`` and ``pruning.ratio`` columns in addition
        to the metrics to visualize.
    metrics : list[str]
        List of metric column names to plot.
    output_dir : str
        Directory where the PNG files will be written.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError:
        # silently ignore if visualization libs are missing
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(["pruning.method", "pruning.ratio"])

    for metric in metrics:
        pivot = grouped[metric].mean().unstack("pruning.ratio")
        plt.figure()
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", cbar_kws={"label": metric})
        plt.xlabel("pruning.ratio")
        plt.ylabel("pruning.method")
        plt.tight_layout()
        safe_metric = metric.replace(".", "_")
        filename = f"{safe_metric}_heatmap.png"
        plt.savefig(out_path / filename)
        plt.close()
