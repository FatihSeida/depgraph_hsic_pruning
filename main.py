"""Run pruning experiments across multiple methods and ratios.

This script iterates over the available pruning methods and a set
of pruning ratios to train and fine-tune models using the
:class:`PruningPipeline` or :class:`PruningPipeline2` depending on the
pruning method. It supports resuming interrupted runs by
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
from pipeline import BasePruningPipeline, PruningPipeline, MonitorComputationStep
import pipeline as pipeline_mod
from ultralytics import YOLO
import prune_methods as pm
from prune_methods import BasePruningMethod
from helper.logger import timed_step, format_subheader, log_block, log_substep


def create_pipeline(
    model_path: str,
    data: str,
    workdir: str,
    method_cls: Type[BasePruningMethod] | None,
    logger: Logger | None = None,
) -> BasePruningPipeline:
    """Create appropriate pipeline based on pruning method type."""
    
    # Methods that rely on the dependency graph implementation
    depgraph_methods = (
        pm.DepgraphHSICMethod,
        pm.DepgraphMethod,
        pm.IsomorphicMethod,
        pm.TorchRandomMethod,
    )
    
    if method_cls in depgraph_methods:
        return pipeline_mod.PruningPipeline2(
            model_path=model_path,
            data=data,
            workdir=workdir,
            pruning_method=None,  # Will be set later
            logger=logger
        )
    else:
        return PruningPipeline(
            model_path=model_path,
            data=data,
            workdir=workdir,
            pruning_method=None,  # Will be set later
            logger=logger
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
                cls = torch.tensor(result, dtype=cls.dtype, device=cls.device).float()  # Convert to float for cdist compatibility
    except Exception:
        logger.exception("aggregate_labels failed")

    # Ensure cls is float for cdist compatibility
    if torch.is_tensor(cls):
        cls = cls.float()

    logger.debug("aggregate_labels output shape=%s", getattr(cls, "shape", None))

    return cls


METHODS_MAP = {
    "l1": "L1NormMethod",
    "random": "RandomMethod",
    "depgraph": "DepgraphMethod",
    "tp_random": "TorchRandomMethod",
    "isomorphic": "IsomorphicMethod",
    "hsic_lasso": "HSICLassoMethod",
    "whc": "WeightedHybridMethod",
    "depgraph_hsic": "DepgraphHSICMethod",
}


def get_method_class(name: str) -> Type[BasePruningMethod]:
    """Return pruning method class by name."""
    return getattr(pm, METHODS_MAP[name])

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
            logger.exception("error checking previous training status")
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
    pruning_scope: str = "backbone",
) -> tuple[BasePruningPipeline, Path]:
    """Run a full pruning pipeline for ``method_cls`` at ``ratio``."""
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "pipeline.log"
    if logger is None:
        # Default to DEBUG logging when no logger is provided so that pipeline
        # executions capture detailed information.
        logger = get_logger(level=logging.DEBUG, log_file=str(log_file))
    else:
        for h in list(logger.logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.logger.removeHandler(h)
        add_file_handler(logger, str(log_file))
    
    # Create pipeline based on method type
    pipeline = create_pipeline(
        model_path=model_path,
        data=data,
        workdir=str(workdir),
        method_cls=method_cls,
        logger=logger
    )
    
    logger.info("%s", "="*80)
    logger.info("🚀 Memulai %s pipeline", "baseline" if method_cls is None else "pretrain")
    logger.info("Model: %s | Metode: %s | Rasio: %.2f", model_path, method_cls.__name__ if method_cls else "baseline", ratio)
    logger.info("Konfigurasi -> baseline_epochs=%d, finetune_epochs=%d, batch_size=%d, device=%s", 
                config.baseline_epochs, config.finetune_epochs, config.batch_size, config.device)
    logger.info("Lokasi kerja: %s", workdir)
    logger.info("%s", "="*80)

    # Load model
    log_block(logger, "LOAD MODEL")
    with timed_step(logger, "Load model"):
        pipeline.load_model()
        if hasattr(pipeline.model, "to"):
            pipeline.model.to(config.device)
            log_substep(logger, f"Model dipindahkan ke device: {config.device}")

    # Calculate initial statistics
    log_block(logger, "HITUNG STATISTIK AWAL")
    with timed_step(logger, "Hitung statistik awal"):
        stats = pipeline.calc_initial_stats()
        log_substep(logger, f"parameters: {stats['parameters']}")
        log_substep(logger, f"flops: {stats['flops']}")
        log_substep(logger, f"filters: {stats['filters']}")
        log_substep(logger, f"model_size_mb: {stats['model_size_mb']}")

    if method_cls is not None:
        # Initialize pruning method with appropriate parameters
        if method_cls == pm.DepgraphHSICMethod:
            pruning_method = method_cls(
                pipeline.model.model, 
                workdir=workdir,
                pruning_scope=pruning_scope
            )
        else:
            pruning_method = method_cls(pipeline.model.model, workdir=workdir)
        
        pipeline.set_pruning_method(pruning_method)
        
        # When skipping pretraining, run analysis immediately so hooks are
        # active for the forthcoming pruning steps.
        if config.baseline_epochs == 0:
            pipeline.analyze_structure()

    if config.baseline_epochs > 0:
        if method_cls is not None:
            # Run analysis before training so hooks can capture activations
            pipeline.analyze_structure()
        monitor = MonitorComputationStep("pretrain")
        with timed_step(logger, "Pretraining/Baseline training"):
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
                    if isinstance(getattr(pipeline, "pruning_method", None), pm.DepgraphHSICMethod)
                    else None
                ),
            )
            mgr = getattr(pipeline, "metrics_mgr", None)
            if mgr is None:
                from helper import MetricManager
                mgr = pipeline.metrics_mgr = MetricManager()
            monitor.stop(mgr)
        # Synthetic data collection is now handled within pipeline.pretrain()

    if method_cls is not None and config.baseline_epochs == 0:
        # Synthetic data collection is handled within pipeline methods when needed
        pass
        
    if method_cls is not None:
        if isinstance(getattr(pipeline, "pruning_method", None), pm.DepgraphHSICMethod):
            # Ensure analysis runs on the latest model state before mask generation
            pipeline.analyze_structure()
        pipeline.generate_pruning_mask(ratio)
        pipeline.apply_pruning()
        snapshot = workdir / "snapshot.pt"
        if hasattr(pipeline, "save_model"):
            pipeline.save_model(snapshot)
        else:
            pipeline.model.save(str(snapshot))
        logger.info("Saved snapshot to %s", snapshot)
        pipeline.model = YOLO(str(snapshot))
        if hasattr(pipeline.model, "to"):
            pipeline.model.to(config.device)
        pruner = getattr(pipeline, "pruning_method", None)
        if pruner is not None:
            try:
                pruner.model = pipeline.model.model
            except Exception:  # pragma: no cover - best effort
                logger.debug("failed to update pruning method model reference")
            if hasattr(pipeline, "_sync_example_inputs_device"):
                try:
                    pipeline._sync_example_inputs_device()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("failed to sync example inputs device")
        final_path = workdir / f"pruned_model_{method_cls.__name__}_{ratio}.pt"
        pipeline.reconfigure_model(output_path=final_path)
        logger.info("Saved pruned model to %s", final_path)
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
                if isinstance(getattr(pipeline, "pruning_method", None), pm.DepgraphHSICMethod)
                else None
            ),
        )
        mgr = getattr(pipeline, "metrics_mgr", None)
        if mgr is None:
            from helper import MetricManager
            mgr = pipeline.metrics_mgr = MetricManager()
        monitor.stop(mgr)
        if method_cls is not None and config.finetune_epochs > 0:
            pipeline.analyze_structure()
            # Synthetic data collection is handled within pipeline methods when needed
    pipeline.visualize_results()
    pipeline.save_pruning_results(workdir / "results.pt")
    if hasattr(pipeline, "save_metrics_csv"):
        csv_path = pipeline.save_metrics_csv(workdir / "metrics.csv")
    else:
        csv_path = pipeline.metrics_mgr.to_csv(workdir / "metrics.csv")
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
        pruning_scope: str = "backbone",
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
        self.pruning_scope = pruning_scope

    def run(self) -> None:
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
                pruning_scope=self.pruning_scope,
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
                    pruning_scope=self.pruning_scope,
                )
                self.manager.add_result(method_name, ratio, pipeline.record_metrics(), csv_path)

        self.manager.compare_pruning_methods()
        # Visualize training metrics across ratios and methods
        for metric in self.metrics:
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
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS_MAP.keys()),
        default=list(METHODS_MAP.keys()),
        help="Pruning methods to evaluate",
    )
    parser.add_argument(
        "--pruning-scope",
        choices=["backbone", "full"],
        default="backbone",
        help="Scope of pruning: 'backbone' for first 10 layers only, 'full' for entire model",
    )
    parser.add_argument("--compare", action="store_true", help="Compare results from previous runs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> None:
    """Replicate the old ``pruning_comparison.py`` behaviour."""
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

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ratio", "run"])

    # Always log at DEBUG level when running comparisons for maximum detail.
    logger = get_logger(level=logging.DEBUG)

    # Baseline training
    baseline_dir: Path | None = None
    baseline_weights = args.model
    if not getattr(args, 'no_baseline', False):
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
            pruning_scope=args.pruning_scope,
        )
        baseline_weights = baseline_dir / "baseline" / "weights" / "best.pt"
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["baseline", 0, baseline_dir.as_posix()])

    for method in args.methods:
        method_cls = get_method_class(method)
        for ratio in args.ratios:
            ratio_tag = str(ratio).replace(".", "_")
            run_dir = base_dir / f"{safe_name(str(method))}_r{ratio_tag}"
            complete_flag = run_dir / ".experiment_complete"
            run_dir.mkdir(parents=True, exist_ok=True)
            if getattr(args, 'cont', False) and complete_flag.exists():
                continue
            config = TrainConfig(
                baseline_epochs=0 if not getattr(args, 'no_baseline', False) else args.baseline_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                ratios=[ratio],
                device=args.device,
            )
            execute_pipeline(
                baseline_weights if not getattr(args, 'no_baseline', False) else args.model,
                args.data,
                method_cls,
                ratio,
                config,
                run_dir,
                resume=args.resume,
                logger=logger,
                pruning_scope=args.pruning_scope,
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
    methods = [get_method_class(m) for m in args.methods]
    # Use DEBUG logging for all runs.  The ``--debug`` flag is retained for
    # backward compatibility but no longer controls the log level.
    logger = get_logger(level=logging.DEBUG)
    runner = ExperimentRunner(
        model_path=args.model,
        data=args.data,
        methods=methods,
        config=config,
        workdir=args.workdir,
        resume=args.resume,
        logger=logger,
        pruning_scope=args.pruning_scope,
    )
    runner.run()


if __name__ == "__main__":
    main()
