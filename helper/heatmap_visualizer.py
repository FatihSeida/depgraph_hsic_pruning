"""Heatmap visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

DEFAULT_METRICS = [
    "pruning.flops.reduction_percent",
    "pruning.parameters.reduction_percent",
    "pruning.model_size_mb.reduction_percent",
]

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
        import matplotlib.artist  # type: ignore  # ensure lazy modules are loaded
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except Exception:
        # silently ignore if visualization libs are missing or misconfigured
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


def plot_default_metric_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """Generate heatmaps for :data:`DEFAULT_METRICS`."""
    plot_metric_heatmaps(df, DEFAULT_METRICS, output_dir)


__all__ = [
    "plot_metric_heatmaps",
    "plot_default_metric_heatmaps",
    "DEFAULT_METRICS",
]
