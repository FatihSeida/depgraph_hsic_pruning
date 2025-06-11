from __future__ import annotations

"""Utilities to manage pruning experiments."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


class ExperimentManager:
    """Manage multiple pruning experiments and produce comparisons."""

    def __init__(self, backbone: str, workdir: str = "runs/experiments") -> None:
        self.backbone = backbone
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def add_result(self, method: str, ratio: float, metrics: Dict[str, Any]) -> None:
        """Record the metrics for a pruning experiment."""
        self.results.append({"method": method, "ratio": ratio, "metrics": metrics})

    def compare_pruning_methods(self) -> None:
        """Visualize mAP against pruning ratio for all experiments."""
        if not self.results:
            return
        ratios = [r["ratio"] for r in self.results]
        maps = [r["metrics"].get("mAP", 0) for r in self.results]
        labels = [r["method"] for r in self.results]
        plt.figure()
        plt.title("Pruning Comparison")
        plt.xlabel("Pruning ratio")
        plt.ylabel("mAP")
        plt.plot(ratios, maps, marker="o")
        for x, y, lbl in zip(ratios, maps, labels):
            plt.text(x, y, lbl)
        plt.tight_layout()
        plt.savefig(self.workdir / "comparison.png")
        plt.close()
