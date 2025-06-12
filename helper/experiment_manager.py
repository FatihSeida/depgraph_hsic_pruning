from __future__ import annotations

"""Utilities to manage pruning experiments."""

from pathlib import Path
from typing import Any, Dict, List


class ExperimentManager:
    """Manage multiple pruning experiments and produce comparisons."""

    def __init__(self, backbone: str, workdir: str = "runs/experiments") -> None:
        self.backbone = backbone
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        method: str,
        ratio: float,
        metrics: Dict[str, Any],
        csv_path: str | Path | None = None,
    ) -> None:
        """Record the metrics and CSV location for a pruning experiment."""

        entry: Dict[str, Any] = {"method": method, "ratio": ratio, "metrics": metrics}
        if csv_path is not None:
            entry["csv"] = str(csv_path)
        self.results.append(entry)

    def compare_pruning_methods(self) -> None:
        """Visualize mAP against pruning ratio for all experiments."""
        if not self.results:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            # matplotlib is optional; skip plotting if unavailable
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

    # ------------------------------------------------------------------
    # Generic plotting helpers
    # ------------------------------------------------------------------
    def _extract_metric(self, data: Dict[str, Any], path: str) -> float | None:
        """Return metric value located at dotted ``path`` inside ``data``."""
        parts = path.split(".")
        value: Any = data
        for p in parts:
            if isinstance(value, dict) and p in value:
                value = value[p]
            else:
                return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def plot_line(self, metric: str) -> None:
        """Plot ``metric`` against ratio for each method."""
        if not self.results:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            return
        method_groups: Dict[str, List[tuple[float, float | None]]] = {}
        for entry in self.results:
            val = self._extract_metric(entry["metrics"], metric)
            method_groups.setdefault(entry["method"], []).append((entry["ratio"], val))
        plt.figure()
        for method, vals in method_groups.items():
            vals.sort(key=lambda x: x[0])
            x = [v[0] for v in vals]
            y = [v[1] if v[1] is not None else 0 for v in vals]
            plt.plot(x, y, marker="o", label=method)
        plt.xlabel("ratio")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        filename = metric.replace(".", "_") + "_line.png"
        plt.savefig(self.workdir / filename)
        plt.close()

    def plot_heatmap(self, metric: str) -> None:
        """Draw heatmap of ``metric`` with methods on x-axis and ratios on y-axis."""
        if not self.results:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
            import seaborn as sns  # type: ignore
        except ImportError:
            return
        methods = sorted({r["method"] for r in self.results})
        ratios = sorted({r["ratio"] for r in self.results})
        matrix = np.full((len(ratios), len(methods)), np.nan)
        for entry in self.results:
            i = ratios.index(entry["ratio"])
            j = methods.index(entry["method"])
            val = self._extract_metric(entry["metrics"], metric)
            if val is not None:
                matrix[i, j] = val
        plt.figure()
        sns.heatmap(
            matrix,
            annot=True,
            cmap="YlGnBu",
            xticklabels=methods,
            yticklabels=[str(r) for r in ratios],
            cbar_kws={"label": metric},
        )
        plt.xlabel("method")
        plt.ylabel("ratio")
        plt.tight_layout()
        filename = metric.replace(".", "_") + "_heatmap.png"
        plt.savefig(self.workdir / filename)
        plt.close()
