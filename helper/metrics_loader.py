"""Load metrics CSV files from experiment runs into a DataFrame."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_metrics_dataframe(root: str | Path) -> pd.DataFrame:
    """Return DataFrame with metrics from all ``metrics.csv`` files under ``root``.

    The DataFrame includes ``pruning.method`` and ``pruning.ratio`` columns
    inferred from the parent directory name of each CSV. Any metrics recorded in
    the CSV are preserved as additional columns.
    """
    root = Path(root)
    records = []
    for csv_path in root.rglob("metrics.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        run_name = csv_path.parent.name
        method = None
        ratio = None
        if "_r" in run_name:
            method, ratio_str = run_name.rsplit("_r", 1)
            ratio_str = ratio_str.replace("_", ".")
            try:
                ratio = float(ratio_str)
            except ValueError:
                ratio = None
        row["pruning.method"] = method
        row["pruning.ratio"] = ratio
        records.append(row)
    return pd.DataFrame(records)

__all__ = ["load_metrics_dataframe"]
