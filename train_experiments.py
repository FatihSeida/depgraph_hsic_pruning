"""Run pruning experiments across multiple methods and ratios.

This script iterates over the available pruning methods and a set
of pruning ratios to train and fine-tune models using the
:class:`PruningPipeline`. It supports resuming interrupted runs by
passing the ``--resume`` flag on the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type

from helper import ExperimentManager, get_logger
from pipeline import PruningPipeline
from prune_methods import (
    BasePruningMethod,
    Method1,
    Method2,
    Method3,
    Method4,
    Method5,
    Method6,
    Method7,
    Method8,
)


@dataclass
class TrainConfig:
    """Configuration parameters for training."""

    baseline_epochs: int = 1
    finetune_epochs: int = 3
    batch_size: int = 16
    ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])


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
    ) -> None:
        self.model_path = model_path
        self.data = data
        self.methods = methods
        self.config = config
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.logger = get_logger()
        self.manager = ExperimentManager(Path(model_path).stem, workdir)

    def run(self) -> None:
        """Execute all pruning experiments."""
        for method_cls in self.methods:
            method_name = method_cls.__name__
            self.logger.info("Running method: %s", method_name)
            for ratio in self.config.ratios:
                run_name = f"{method_name}_r{ratio}"
                run_dir = self.workdir / run_name
                pipeline = PruningPipeline(
                    self.model_path,
                    data=self.data,
                    workdir=str(run_dir),
                    logger=self.logger,
                )
                pipeline.load_model()
                pipeline.set_pruning_method(method_cls(pipeline.model.model, workdir=run_dir))
                pipeline.calc_initial_stats()

                pipeline.pretrain(
                    epochs=self.config.baseline_epochs,
                    batch=self.config.batch_size,
                    project=str(run_dir),
                    name="pretrain",
                    resume=self.resume,
                )

                pipeline.analyze_structure()
                pipeline.generate_pruning_mask(ratio)
                pipeline.apply_pruning()
                pipeline.reconfigure_model()
                pipeline.calc_pruned_stats()

                pipeline.finetune(
                    epochs=self.config.finetune_epochs,
                    batch=self.config.batch_size,
                    project=str(run_dir),
                    name="finetune",
                    resume=self.resume,
                )

                self.manager.add_result(method_name, ratio, pipeline.record_metrics())
                pipeline.visualize_results()
                pipeline.save_pruning_results(run_dir / "results")

        self.manager.compare_pruning_methods()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run multiple pruning experiments")
    parser.add_argument("--model", required=True, help="Path to the model weights")
    parser.add_argument("--data", required=True, help="Dataset YAML describing train/val paths")
    parser.add_argument("--workdir", default="runs/experiments", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted runs if possible")
    parser.add_argument("--baseline-epochs", type=int, default=1, help="Number of pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Number of finetuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Pruning ratios to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        baseline_epochs=args.baseline_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        ratios=args.ratios,
    )
    methods = [Method1, Method2, Method3, Method4, Method5, Method6, Method7, Method8]
    runner = ExperimentRunner(
        args.model,
        args.data,
        methods,
        config,
        workdir=args.workdir,
        resume=args.resume,
    )
    runner.run()


if __name__ == "__main__":
    main()
