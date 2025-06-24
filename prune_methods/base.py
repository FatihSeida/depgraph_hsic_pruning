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

    def __init__(
        self,
        model: Any,
        workdir: str | Path = "runs/pruning",
        example_inputs: torch.Tensor | tuple | None = None,
    ) -> None:
        self.model = model
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.initial_stats: Dict[str, float] = {}
        self.pruned_stats: Dict[str, float] = {}
        self.logger: Logger = get_logger()
        self.masks: List[torch.Tensor] = []
        self.example_inputs = example_inputs or torch.randn(1, 3, 640, 640)

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

    # ------------------------------------------------------------------
    # Optional visualization and saving methods
    # ------------------------------------------------------------------
    def visualize_comparison(self) -> None:
        """Visualize baseline vs pruned metrics comparison.

        The default implementation generates a simple bar chart comparing
        the values stored in :attr:`initial_stats` and :attr:`pruned_stats`.
        If plotting libraries are unavailable the method fails silently and a
        warning is logged instead of raising an exception.
        """

        if not self.initial_stats or not self.pruned_stats:
            return

        try:  # optional dependency
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - plotting optional
            self.logger.warning("Failed to import matplotlib: %s", exc)
            return

        try:
            metrics = sorted(set(self.initial_stats) | set(self.pruned_stats))
            baseline = [self.initial_stats.get(m, 0) for m in metrics]
            pruned = [self.pruned_stats.get(m, 0) for m in metrics]

            fig, ax = plt.subplots(figsize=(8, 4))
            x = range(len(metrics))
            width = 0.35
            ax.bar([i - width / 2 for i in x], baseline, width=width, label="initial")
            ax.bar([i + width / 2 for i in x], pruned, width=width, label="pruned")
            ax.set_xticks(list(x))
            ax.set_xticklabels(metrics, rotation=45, ha="right")
            ax.legend()
            fig.tight_layout()
            plt.savefig(self.workdir / "comparison.png")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - plotting optional
            self.logger.warning("Failed to create comparison plot: %s", exc)

    def visualize_pruned_filters(self) -> None:
        """Visualize which filters were pruned.

        When :attr:`masks` is populated a heatmap is produced illustrating the
        pruned channels for every layer.  Layers are placed on the y-axis and
        channel indices on the x-axis.  Unavailable plotting libraries are
        tolerated and only result in a warning.
        """

        if not self.masks:
            return

        try:  # optional dependency
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to import plotting libs: %s", exc)
            return

        try:
            lengths = [len(m) for m in self.masks]
            max_channels = max(lengths)
            matrix = np.zeros((len(self.masks), max_channels), dtype=int)
            for i, mask in enumerate(self.masks):
                arr = (~mask).cpu().numpy().astype(int)
                matrix[i, : len(arr)] = arr

            fig, ax = plt.subplots(figsize=(8, 0.5 * len(self.masks) + 1))
            ax.imshow(matrix, cmap="Greys", aspect="auto")
            ax.set_ylabel("Layer")
            ax.set_xlabel("Channel index")
            ax.set_title("Pruned filter map (dark = pruned)")
            plt.tight_layout()
            plt.savefig(self.workdir / "pruned_filters.png")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - plotting optional
            self.logger.warning("Failed to create pruned filter map: %s", exc)

    def save_results(self, path: str | Path) -> None:
        """Save pruning results to the specified path.

        A pickle file is written containing the attributes ``masks``,
        ``initial_stats`` and ``pruned_stats``.  Any I/O error is caught and
        logged without raising an exception.
        """

        try:
            import pickle

            data = {
                "masks": self.masks,
                "initial_stats": self.initial_stats,
                "pruned_stats": self.pruned_stats,
            }

            path = Path(path)
            with path.open("wb") as f:
                pickle.dump(data, f)

            self.logger.info("Pruning results saved to %s", path)
        except Exception as exc:  # pragma: no cover - optional
            self.logger.warning("Failed to save pruning results: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _inputs_tuple(self) -> tuple:
        """Return ``example_inputs`` as a tuple on the model's device."""
        inputs = self.example_inputs
        if not isinstance(inputs, tuple):
            if isinstance(inputs, list):
                inputs = tuple(inputs)
            else:
                inputs = (inputs,)
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        moved = []
        for t in inputs:
            if torch.is_tensor(t):
                moved.append(t.to(device))
            else:
                moved.append(t)
        return tuple(moved)

