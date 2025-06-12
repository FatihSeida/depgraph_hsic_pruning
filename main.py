"""Run pruning experiments across multiple methods and ratios.

This script iterates over the available pruning methods and a set
of pruning ratios to train and fine-tune models using the
:class:`PruningPipeline`. It supports resuming interrupted runs by
passing the ``--resume`` flag on the command line.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Type
import logging

from helper import ExperimentManager, get_logger, Logger
from pipeline import PruningPipeline
from prune_methods import (
    BasePruningMethod,
    L1NormPruningMethod,
    RandomPruningMethod,
)


METHODS_MAP = {
    "l1": L1NormPruningMethod,
    "random": RandomPruningMethod,
}


def safe_name(value: str) -> str:
    """Return a filesystem-friendly version of ``value``."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


@dataclass
class TrainConfig:
    """Configuration parameters for training."""

    baseline_epochs: int = 1
    finetune_epochs: int = 3
    batch_size: int = 16
    ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    device: str | int | list = 0


def execute_pipeline(
    model_path: str,
    data: str,
    method_cls: Type[BasePruningMethod] | None,
    ratio: float,
    config: TrainConfig,
    workdir: Path,
    *,
    resume: bool = False,
    logger=None,
) -> PruningPipeline:
    """Run a full pruning pipeline for ``method_cls`` at ``ratio``."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "pipeline.log"
    if logger is None:
        logger = get_logger(log_file=str(log_file))
    else:
        get_logger(log_file=str(log_file))
    pipeline = PruningPipeline(model_path, data=data, workdir=str(workdir), logger=logger)
    pipeline.load_model()
    if method_cls is not None:
        pipeline.set_pruning_method(method_cls(pipeline.model.model, workdir=workdir))
    pipeline.calc_initial_stats()
    pipeline.pretrain(
        epochs=config.baseline_epochs,
        batch=config.batch_size,
        project=str(workdir),
        name="baseline" if method_cls is None else "pretrain",
        resume=resume,
        device=config.device,
    )
    if method_cls is not None:
        pipeline.analyze_structure()
        pipeline.generate_pruning_mask(ratio)
        pipeline.apply_pruning()
        pipeline.reconfigure_model()
        pipeline.calc_pruned_stats()
        pipeline.finetune(
            epochs=config.finetune_epochs,
            batch=config.batch_size,
            project=str(workdir),
            name="finetune",
            resume=resume,
            device=config.device,
        )
    pipeline.visualize_results()
    pipeline.save_pruning_results(workdir / "results")
    return pipeline


class ExperimentRunner:
    """Orchestrate pruning experiments for multiple methods."""

    def __init__(
        self,
        model_path: str,
        data: str,
        methods: List[Type[BasePruningMethod]],
        config: TrainConfig,
        workdir: str = "runs/experiments",
        *,
        resume: bool = False,
        logger: Logger | None = None,
    ) -> None:
        self.model_path = model_path
        self.data = data
        self.methods = methods
        self.config = config
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.logger = logger or get_logger()
        self.manager = ExperimentManager(Path(model_path).stem, workdir)

    def run(self) -> None:
        """Execute all pruning experiments."""
        for method_cls in self.methods:
            method_name = method_cls.__name__
            self.logger.info("Running method: %s", method_name)
            for ratio in self.config.ratios:
                run_name = f"{method_name}_r{ratio}"
                run_dir = self.workdir / run_name
                pipeline = execute_pipeline(
                    self.model_path,
                    self.data,
                    method_cls,
                    ratio,
                    self.config,
                    run_dir,
                    resume=self.resume,
                    logger=self.logger,
                )
                self.manager.add_result(method_name, ratio, pipeline.record_metrics())

        self.manager.compare_pruning_methods()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run multiple pruning experiments")
    parser.add_argument("--model", required=True, help="Path to the model weights")
    parser.add_argument(
        "--data",
        default="biotech_model_train.yaml",
        help="Dataset YAML describing train/val paths",
    )
    parser.add_argument("--workdir", default="runs/experiments", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted runs if possible")
    parser.add_argument("--baseline-epochs", type=int, default=1, help="Number of pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Number of finetuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--device", default="cuda:0", help="Training device for YOLO")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Pruning ratios to evaluate",
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS_MAP.keys()), help="Pruning methods to evaluate")
    parser.add_argument("--runs-dir", default="experiments", help="Root directory for comparison runs")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline training in comparison mode")
    parser.add_argument("--debug", action="store_true", help="Enable verbose output")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Continue incomplete runs")
    parser.add_argument("--compare", action="store_true", help="Run comparison across methods")
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> None:
    """Replicate the old ``pruning_comparison.py`` behaviour."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = repo_root / args.runs_dir / f"pruning_comparison_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "comparison_report.csv"
    status_file = base_dir / "experiment_status.txt"
    vis_dir = base_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with status_file.open("w") as f:
        f.write("running\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ratio", "run"])

    logger = get_logger()

    # Baseline training
    baseline_dir: Path | None = None
    if not args.no_baseline:
        baseline_dir = base_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        config = TrainConfig(
            baseline_epochs=args.baseline_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            ratios=[0],
            device=args.device,
        )
        execute_pipeline(
            args.model,
            args.data,
            None,
            0,
            config,
            baseline_dir,
            resume=args.resume,
            logger=logger,
        )
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["baseline", 0, baseline_dir.as_posix()])

    for method in args.methods:
        method_cls = METHODS_MAP[method]
        for ratio in args.ratios:
            ratio_tag = str(ratio).replace(".", "_")
            run_dir = base_dir / f"{safe_name(str(method))}_r{ratio_tag}"
            complete_flag = run_dir / ".experiment_complete"
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.cont and complete_flag.exists():
                continue
            config = TrainConfig(
                baseline_epochs=args.baseline_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                ratios=[ratio],
                device=args.device,
            )
            execute_pipeline(
                args.model,
                args.data,
                method_cls,
                ratio,
                config,
                run_dir,
                resume=args.resume,
                logger=logger,
            )
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([method, ratio, run_dir.as_posix()])
            complete_flag.touch()

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


def main() -> None:
    args = parse_args()
    if args.compare:
        run_comparison(args)
        return

    config = TrainConfig(
        baseline_epochs=args.baseline_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        ratios=args.ratios,
        device=args.device,
    )
    methods = [METHODS_MAP[m] for m in args.methods]
    logger = get_logger(level=logging.DEBUG if args.debug else logging.INFO)
    runner = ExperimentRunner(
        args.model,
        args.data,
        methods,
        config,
        workdir=args.workdir,
        resume=args.resume,
        logger=logger,
    )
    runner.run()


if __name__ == "__main__":
    main()
