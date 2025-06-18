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

from helper import ExperimentManager, get_logger, Logger, add_file_handler
from pipeline import PruningPipeline, MonitorComputationStep
from prune_methods import (
    BasePruningMethod,
    L1NormMethod,
    RandomMethod,
    DepgraphMethod,
    TorchRandomMethod,
    IsomorphicMethod,
    HSICLassoMethod,
    DepgraphHSICMethod,
    WeightedHybridMethod,
)


def aggregate_labels(batch):
    """Return a representative label for each image.

    Detection batches often provide one label per object. When the number of
    labels differs from the batch size this function collapses them to a single
    label per image using a majority vote. ``batch['batch_idx']`` is used to map
    objects to images when available.
    """

    cls = batch.get("cls")
    img = batch.get("img")
    batch_idx = batch.get("batch_idx")
    logger = logging.getLogger("pruning")
    logger.debug(
        "aggregate_labels input cls shape=%s batch_idx shape=%s",
        getattr(cls, "shape", None),
        getattr(batch_idx, "shape", None),
    )

    try:
        import torch
        if torch.is_tensor(cls) and torch.is_tensor(img) and len(cls) != img.shape[0]:
            logger.debug("majority vote aggregation")
            cls = cls.view(-1).long()
            bs = img.shape[0]
            result = []

            if torch.is_tensor(batch_idx) and len(batch_idx) == len(cls):
                batch_idx = batch_idx.view(-1)
                for i in range(bs):
                    labels_i = cls[batch_idx == i]
                    if labels_i.numel() == 0:
                        continue
                    result.append(labels_i.mode().values.item())
            else:
                cls = cls.view(bs, -1)
                for labels_i in cls:
                    result.append(labels_i.mode().values.item())

            if result:
                cls = torch.tensor(result, dtype=cls.dtype, device=cls.device)
    except Exception:
        pass

    logger.debug("aggregate_labels output shape=%s", getattr(cls, "shape", None))

    return cls


METHODS_MAP = {
    "l1": L1NormMethod,
    "random": RandomMethod,
    "depgraph": DepgraphMethod,
    "tp_random": TorchRandomMethod,
    "isomorphic": IsomorphicMethod,
    "hsic_lasso": HSICLassoMethod,
    "whc": WeightedHybridMethod,
    "depgraph_hsic": DepgraphHSICMethod,
}

# Default metrics visualized when no custom list is provided
DEFAULT_PLOT_METRICS = [
    "pruning.flops.reduction_percent",
    "pruning.filters.reduction_percent",
    "computation.total_time_minutes",
    "training.recall",
    "training.mAP50_95",
    "pruning.parameters.reduction_percent",
    "training.precision",
    "training.mAP",
    "pruning.model_size_mb.reduction_percent",
    "computation.avg_ram_used_mb",
]


def safe_name(value: str) -> str:
    """Return a filesystem-friendly version of ``value``."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


@dataclass
class TrainConfig:
    """Configuration parameters for training.

    ``reuse_baseline`` controls whether the baseline weights should be loaded
    from the previous run if available. When ``True`` and a ``best.pt`` file is
    found in the baseline directory, pretraining is skipped and the weights are
    reused.
    """

    baseline_epochs: int = 1
    finetune_epochs: int = 3
    batch_size: int = 16
    ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    device: str | int | list = 0
    reuse_baseline: bool = True


def _adjust_resume(phase_dir: Path, epochs: int, resume: bool, logger: Logger) -> bool:
    """Return ``resume`` unless previous training completed."""

    if not resume:
        return resume
    last_pt = phase_dir / "weights" / "last.pt"
    best_pt = phase_dir / "weights" / "best.pt"
    results = phase_dir / "results.csv"
    if not last_pt.exists():
        logger.info("training completed previously – starting a new run")
        return False
    if best_pt.exists() and results.exists():
        try:
            import csv

            with results.open() as f:
                rows = list(csv.DictReader(f))
            if rows:
                final_epoch = int(rows[-1].get("epoch", -1)) + 1
                if final_epoch >= epochs:
                    logger.info("training completed previously – starting a new run")
                    return False
        except Exception:
            pass
    return resume


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
) -> tuple[PruningPipeline, Path]:
    """Run a full pruning pipeline for ``method_cls`` at ``ratio``."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "pipeline.log"
    if logger is None:
        logger = get_logger(log_file=str(log_file))
    else:
        add_file_handler(logger, str(log_file))
    pipeline = PruningPipeline(model_path, data=data, workdir=str(workdir), logger=logger)
    pipeline.load_model()
    pipeline.calc_initial_stats()
    if method_cls is not None:
        pipeline.set_pruning_method(method_cls(pipeline.model.model, workdir=workdir))
        pipeline.analyze_structure()

    if config.baseline_epochs > 0:
        monitor = MonitorComputationStep("pretrain")
        monitor.start()
        phase = "baseline" if method_cls is None else "pretrain"
        pretrain_resume = _adjust_resume(workdir / phase, config.baseline_epochs, resume, logger)
        pipeline.pretrain(
            epochs=config.baseline_epochs,
            batch=config.batch_size,
            project=str(workdir),
            name=phase,
            resume=pretrain_resume,
            device=config.device,
            label_fn=(
                aggregate_labels
                if isinstance(getattr(pipeline, "pruning_method", None), DepgraphHSICMethod)
                else None
            ),
        )
        mgr = getattr(pipeline, "metrics_mgr", None)
        if mgr is None:
            from helper import MetricManager
            mgr = pipeline.metrics_mgr = MetricManager()
        monitor.stop(mgr)

    if method_cls is not None and config.baseline_epochs == 0:
        pipeline.analyze_structure()
        if (
            isinstance(getattr(pipeline, "pruning_method", None), DepgraphHSICMethod)
            and config.reuse_baseline
            and config.baseline_epochs == 0
        ):
            try:
                import yaml
                import torch
                from pathlib import Path

                with open(data) as f:
                    ds_cfg = yaml.safe_load(f)
                base = Path(ds_cfg.get("path", Path(data).parent))
                val = ds_cfg.get("val") or ds_cfg.get("train")
                val_path = base / val if val is not None else base
                if val_path.is_dir():
                    imgs = sorted(val_path.glob("*.*"))
                    img = imgs[0] if imgs else None
                else:
                    img = val_path if val_path.exists() else None
                if img is not None:
                    logger.info("short forward pass image: %s", img)
                    label_file = Path(str(img)).with_suffix(".txt").as_posix()
                    label_file = label_file.replace("/images/", "/labels/")
                    logger.info("short forward pass label file: %s", label_file)
                    y = torch.tensor([])
                    lf = Path(label_file)
                    if lf.exists():
                        with lf.open() as lf_f:
                            labels = [float(line.split()[0]) for line in lf_f if line.strip()]
                        if labels:
                            y = torch.tensor([labels[0]])
                            logger.info("label file %s has %d entries", label_file, len(labels))
                        else:
                            logger.warning("label file %s is empty", label_file)
                    else:
                        logger.warning("label file %s does not exist", label_file)
                    if y.numel() > 0:
                        try:
                            from PIL import Image  # type: ignore
                            import numpy as np  # type: ignore

                            img_pil = Image.open(img).convert("RGB")
                            orig_size = img_pil.size
                            if orig_size != (640, 640):
                                logger.debug(
                                    "resizing short forward pass image from %s to (640, 640)",
                                    orig_size,
                                )
                                img_pil = img_pil.resize((640, 640))
                            arr = np.array(img_pil, dtype=np.float32)
                            arr = np.transpose(arr, (2, 0, 1))
                            inp = torch.tensor(arr).unsqueeze(0)
                            logger.debug("short forward pass tensor shape: %s", tuple(inp.shape))
                        except Exception:  # pragma: no cover - fallback when PIL is missing
                            inp = getattr(
                                pipeline.pruning_method,
                                "example_inputs",
                                torch.randn(1, 3, 640, 640),
                            )

                        device = next(pipeline.model.model.parameters()).device
                        with torch.no_grad():
                            pipeline.model.model(inp.to(device))
                        pipeline.pruning_method.add_labels(y)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("short forward pass failed: %s", exc)
    if method_cls is not None:
        pipeline.generate_pruning_mask(ratio)
        pipeline.apply_pruning()
        pipeline.reconfigure_model()
        pipeline.calc_pruned_stats()
        monitor = MonitorComputationStep("finetune")
        monitor.start()
        finetune_resume = _adjust_resume(workdir / "finetune", config.finetune_epochs, resume, logger)
        pipeline.finetune(
            epochs=config.finetune_epochs,
            batch=config.batch_size,
            project=str(workdir),
            name="finetune",
            resume=finetune_resume,
            device=config.device,
            label_fn=(
                aggregate_labels
                if isinstance(getattr(pipeline, "pruning_method", None), DepgraphHSICMethod)
                else None
            ),
        )
        mgr = getattr(pipeline, "metrics_mgr", None)
        if mgr is None:
            from helper import MetricManager
            mgr = pipeline.metrics_mgr = MetricManager()
        monitor.stop(mgr)
    pipeline.visualize_results()
    pipeline.save_pruning_results(workdir / "results.pt")
    csv_path = pipeline.save_metrics_csv(workdir / "metrics.csv")
    return pipeline, csv_path


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
        metrics: List[str] | None = None,
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
        self.metrics = metrics or DEFAULT_PLOT_METRICS

    def run(self, heatmap_only: bool = False) -> None:
        """Execute all pruning experiments."""
        baseline_dir = self.workdir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        weights_file = baseline_dir / "baseline" / "weights" / "best.pt"

        if self.config.reuse_baseline and weights_file.exists():
            baseline_weights = weights_file
        else:
            baseline_weights = self.model_path
            base_cfg = TrainConfig(
                baseline_epochs=self.config.baseline_epochs,
                finetune_epochs=self.config.finetune_epochs,
                batch_size=self.config.batch_size,
                ratios=[0],
                device=self.config.device,
            )
            execute_pipeline(
                self.model_path,
                self.data,
                None,
                0,
                base_cfg,
                baseline_dir,
                resume=self.resume,
                logger=self.logger,
            )
            baseline_weights = weights_file

        for method_cls in self.methods:
            method_name = method_cls.__name__
            self.logger.info("Running method: %s", method_name)
            for ratio in self.config.ratios:
                run_name = f"{method_name}_r{ratio}"
                run_dir = self.workdir / run_name
                pipeline, csv_path = execute_pipeline(
                    baseline_weights,
                    self.data,
                    method_cls,
                    ratio,
                    TrainConfig(
                        baseline_epochs=0,
                        finetune_epochs=self.config.finetune_epochs,
                        batch_size=self.config.batch_size,
                        ratios=[ratio],
                        device=self.config.device,
                    ),
                    run_dir,
                    resume=self.resume,
                    logger=self.logger,
                )
                self.manager.add_result(method_name, ratio, pipeline.record_metrics(), csv_path)

        self.manager.compare_pruning_methods()
        # Visualize training metrics across ratios and methods
        for metric in self.metrics:
            if heatmap_only:
                self.manager.plot_heatmap(metric)
            else:
                self.manager.plot_line(metric)
                self.manager.plot_heatmap(metric)


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
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Only generate heatmap visualizations",
    )
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
    baseline_weights = args.model
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
        baseline_weights = baseline_dir / "baseline" / "weights" / "best.pt"
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
                baseline_epochs=0 if not args.no_baseline else args.baseline_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                ratios=[ratio],
                device=args.device,
            )
            execute_pipeline(
                baseline_weights if not args.no_baseline else args.model,
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
    runner.run(heatmap_only=args.heatmap_only)


if __name__ == "__main__":
    main()
