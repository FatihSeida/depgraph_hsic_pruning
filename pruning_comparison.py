#!/usr/bin/env python3
"""Run a series of pruning experiments and produce a comparison report."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

# ---------------------------------------------------------------------------
# Default configuration constants
# ---------------------------------------------------------------------------
MODEL = "yolov8n.pt"
DATA = "coco8.yaml"
DEVICE = "cpu"
RUNS_DIR = "experiments"
METHODS = ["l1", "random"]
RATIOS = [0.2, 0.4, 0.6, 0.8]
BASELINE_EPOCHS = 1


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def safe_name(value: str) -> str:
    """Return a filesystem-friendly version of ``value``."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pruning methods")
    parser.add_argument("--model", default=MODEL, help="Path to model weights")
    parser.add_argument("--data", default=DATA, help="Dataset YAML")
    parser.add_argument("--device", default=DEVICE, help="Computation device")
    parser.add_argument("--runs-dir", default=RUNS_DIR, help="Root directory for experiments")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Pruning methods to evaluate")
    parser.add_argument("--ratios", nargs="+", type=float, default=RATIOS, help="Prune ratios")
    parser.add_argument("--baseline-epochs", type=int, default=BASELINE_EPOCHS, help="Baseline training epochs")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline training")
    parser.add_argument("--debug", action="store_true", help="Enable verbose output")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Continue incomplete runs")
    return parser.parse_args()


def run_command(cmd: Iterable[str], debug: bool = False) -> None:
    if debug:
        print("Running:", " ".join(cmd))
    subprocess.run(list(cmd), check=False)


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = repo_root / "experiments" / f"pruning_comparison_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "comparison_report.csv"
    status_file = base_dir / "experiment_status.txt"
    vis_dir = base_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with status_file.open("w") as f:
        f.write("running\n")

    # Baseline training -------------------------------------------------
    baseline_dir: Path | None = None
    if not args.no_baseline:
        baseline_dir = base_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            "main.py",
            "--pruning-method",
            "baseline",
            "--prune-ratio",
            "0",
            "--model",
            args.model,
            "--data",
            args.data,
            "--device",
            args.device,
            "--epochs",
            str(args.baseline_epochs),
            "--project",
            str(baseline_dir),
        ]
        run_command(cmd, debug=args.debug)
        (base_dir / "run_name.txt").write_text(baseline_dir.name)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "ratio", "run"])
            writer.writerow(["baseline", 0, baseline_dir.as_posix()])

    # Pruning experiments -----------------------------------------------
    for method in args.methods:
        for ratio in args.ratios:
            safe_method = safe_name(str(method))
            ratio_tag = str(ratio).replace(".", "_")
            run_dir = base_dir / f"{safe_method}_r{ratio_tag}"
            complete_flag = run_dir / ".experiment_complete"
            if args.cont and complete_flag.exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python",
                "main.py",
                "--pruning-method",
                str(method),
                "--prune-ratio",
                str(ratio),
                "--model",
                args.model,
                "--data",
                args.data,
                "--device",
                args.device,
                "--project",
                str(run_dir),
            ]
            run_command(cmd, debug=args.debug)
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([method, ratio, run_dir.as_posix()])
            complete_flag.touch()

    # Placeholder for global visualisations -----------------------------
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        if not df.empty:
            plt.figure()
            for method, group in df.groupby("method"):
                if method == "baseline":
                    continue
                plt.plot(group["ratio"], [1] * len(group), label=method)
            plt.legend()
            plt.xlabel("Prune ratio")
            plt.ylabel("mAP")
            plt.tight_layout()
            plt.savefig(vis_dir / "map_vs_ratio.png")
            plt.close()
    except Exception as exc:  # pragma: no cover - optional dependency
        if args.debug:
            print("Visualization failed:", exc)

    with status_file.open("w") as f:
        f.write("complete\n")

    print("Baseline model:", baseline_dir)
    print("Report CSV:", csv_path)
    print("Visualizations:", vis_dir)


if __name__ == "__main__":
    main()
