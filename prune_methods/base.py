"""Base classes for model pruning methods.

This module defines an abstract :class:`BasePruningMethod` that encapsulates the
common interface used by all pruning algorithms.  Individual pruning strategies
should subclass :class:`BasePruningMethod` and implement the analysis,
mask generation and pruning application steps.

Additionally utility hooks for visualising and storing pruning results are
provided.  The default implementations only act as placeholders and should be
extended by concrete subclasses.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List

import torch

import matplotlib.pyplot as plt

from helper import get_logger, Logger


class BasePruningMethod(abc.ABC):
    """Abstract interface for pruning algorithms."""

    requires_reconfiguration: bool = True

    def __init__(self, model: Any, workdir: str | Path = "runs/pruning") -> None:
        self.model = model
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.initial_stats: Dict[str, float] = {}
        self.pruned_stats: Dict[str, float] = {}
        self.logger: Logger = get_logger()
        self.masks: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Core pruning steps to be implemented by subclasses
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def analyze_model(self) -> None:
        """Inspect model structure and gather information for pruning."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_pruning_mask(self, ratio: float, dataloader=None) -> None:
        """Create a pruning mask with the given sparsity ``ratio``."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_pruning(self, rebuild=False) -> None:
        """Apply the previously generated pruning mask to ``self.model``."""
        raise NotImplementedError