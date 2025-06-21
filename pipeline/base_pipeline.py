from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict

from prune_methods.base import BasePruningMethod
from helper import get_logger, Logger


class BasePruningPipeline(abc.ABC):
    """Base class for building custom pruning pipelines.

    Each method is an extension point that lets you integrate specific
    pruning logic or training code.  Subclasses are expected to override all
    abstract methods to provide concrete behaviour.  The default
    implementation only stores bookkeeping information.
    """

    def __init__(
        self,
        model_path: str,
        data: str,
        workdir: str = "runs/pruning",
        pruning_method: BasePruningMethod | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.model_path = model_path
        self.data = data
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.pruning_method = pruning_method
        self.logger = logger or get_logger()
        self.initial_stats: Dict[str, float] = {}
        self.pruned_stats: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}

    def set_pruning_method(self, method: BasePruningMethod) -> None:
        """Attach a :class:`BasePruningMethod` instance to the pipeline."""
        self.logger.info("Setting pruning method: %s", method.__class__.__name__)
        self.pruning_method = method

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the model located at :pyattr:`self.model_path`."""

    @abc.abstractmethod
    def calc_initial_stats(self) -> Dict[str, float]:
        """Return parameter count and FLOPs before pruning."""

    @abc.abstractmethod
    def pretrain(self, **train_kwargs: Any) -> Dict[str, Any]:
        """Optionally pretrain the model and record metrics."""

    @abc.abstractmethod
    def analyze_structure(self) -> None:
        """Analyze model structure to guide pruning decisions."""

    @abc.abstractmethod
    def generate_pruning_mask(self, ratio: float, dataloader: Any | None = None) -> None:
        """Create a pruning mask with the given sparsity ``ratio`` using ``dataloader`` if provided."""

    @abc.abstractmethod
    def apply_pruning(self) -> None:
        """Apply the previously generated pruning mask."""

    @abc.abstractmethod
    def reconfigure_model(self) -> None:
        """Reconfigure the model after pruning, if necessary."""

    @abc.abstractmethod
    def calc_pruned_stats(self) -> Dict[str, float]:
        """Return parameter count and FLOPs after pruning."""

    @abc.abstractmethod
    def finetune(self, **train_kwargs: Any) -> Dict[str, Any]:
        """Finetune the pruned model and record metrics."""

    def record_metrics(self) -> Dict[str, Any]:
        """Return accumulated training and pruning metrics."""
        return {
            "initial": self.initial_stats,
            "pruned": self.pruned_stats,
            "training": self.metrics,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers around the pruning method
    # ------------------------------------------------------------------
    def visualize_results(self) -> None:
        """Produce plots comparing the baseline and pruned model."""
        if self.pruning_method is not None:
            self.logger.info("Visualizing pruning results")
            self.pruning_method.visualize_comparison()
            self.pruning_method.visualize_pruned_filters()

    def save_pruning_results(self, path: str | Path) -> None:
        """Delegate result saving to the active pruning method."""
        if self.pruning_method is not None:
            self.logger.info("Saving pruning results to %s", path)
            self.pruning_method.save_results(path)

