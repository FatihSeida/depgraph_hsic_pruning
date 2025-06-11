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
from typing import Any, Dict


class BasePruningMethod(abc.ABC):
    """Abstract interface for pruning algorithms."""

    def __init__(self, model: Any, workdir: str | Path = "runs/pruning") -> None:
        self.model = model
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.initial_stats: Dict[str, float] = {}
        self.pruned_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core pruning steps to be implemented by subclasses
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def analyze_model(self) -> None:
        """Inspect model structure and gather information for pruning."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_pruning_mask(self, ratio: float) -> None:
        """Create a pruning mask with the given sparsity ``ratio``."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_pruning(self) -> None:
        """Apply the previously generated pruning mask to ``self.model``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility hooks
    # ------------------------------------------------------------------
    def visualize_comparison(self) -> None:
        """Visualize baseline vs pruned metrics.

        Implementations should compare FLOPs, parameter count, number of
        filters and file size between the unpruned and pruned models.  The
        default implementation is empty.
        """

        pass

    def visualize_pruned_filters(self) -> None:
        """Visualize which filters were removed by pruning."""
        pass

    def save_results(self, path: str | Path) -> None:
        """Persist pruning results to ``path``.

        Subclasses may store masks, statistics or other artefacts useful for
        reproducing the pruning procedure.
        """

        pass
