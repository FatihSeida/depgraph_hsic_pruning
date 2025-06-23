from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict

import torch

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
        self._label_callback = None

    def set_pruning_method(self, method: BasePruningMethod) -> None:
        """Attach a :class:`BasePruningMethod` instance to the pipeline."""
        self.logger.info("Setting pruning method: %s", method.__class__.__name__)
        self.pruning_method = method

    # ------------------------------------------------------------------
    # Common functionality from both pipelines
    # ------------------------------------------------------------------
    def _register_label_callback(self, label_fn) -> None:
        """Attach a callback to record labels for DepgraphHSICMethod."""
        try:
            from prune_methods.depgraph_hsic import DepgraphHSICMethod
        except Exception:
            DepgraphHSICMethod = None

        if DepgraphHSICMethod is None or not isinstance(self.pruning_method, DepgraphHSICMethod):
            return

        if self._label_callback is None:
            def record_labels(trainer) -> None:
                batch = getattr(trainer, "batch", None)
                if isinstance(batch, dict) and "cls" in batch:
                    labels = label_fn(batch)
                    try:
                        import torch
                        if torch.is_tensor(labels) and len(labels) != batch["img"].shape[0]:
                            self.logger.warning(
                                "label_fn returned %d labels for batch size %d; labels may be mismatched",
                                len(labels),
                                batch["img"].shape[0],
                            )
                    except Exception:
                        pass
                    self.logger.debug(
                        "Adding labels for batch with shape %s", tuple(getattr(labels, "shape", []))
                    )
                    self.pruning_method.add_labels(labels)

            self._label_callback = record_labels

        try:
            existing = getattr(self.model, "callbacks", {}).get("on_train_batch_end", [])
            if self._label_callback not in existing:
                self.model.add_callback("on_train_batch_end", self._label_callback)
        except AttributeError:
            pass

    def _unregister_label_callback(self) -> None:
        """Remove the label recording callback if present."""
        try:
            callbacks = getattr(self.model, "callbacks", {}).get("on_train_batch_end", [])
            if self._label_callback in callbacks:
                callbacks.remove(self._label_callback)
        except Exception:
            pass
        self._label_callback = None

    def _sync_pruning_method(self, reanalyze: bool = False) -> None:
        """Synchronize pruning method with the current model."""
        if self.pruning_method is None or self.model is None:
            return
        self.pruning_method.model = self.model.model
        if reanalyze:
            self._sync_example_inputs_device()
            self.pruning_method.analyze_model()

    def _sync_example_inputs_device(self) -> None:
        """Move example_inputs to the current model's device if needed."""
        if not (
            hasattr(self.pruning_method, "example_inputs")
            and self.model is not None
        ):
            return
        try:
            import torch
            device = next(self.model.model.parameters()).device
            ex_inputs = getattr(self.pruning_method, "example_inputs")
            if torch.is_tensor(ex_inputs):
                self.pruning_method.example_inputs = ex_inputs.to(device)
                ex_device = self.pruning_method.example_inputs.device
            self.logger.debug(
                "Model device: %s, example_inputs device: %s", device, ex_device
            )
        except Exception:
            pass

    def _collect_synthetic_activations(self, num_samples=4) -> int:
        """Collect activations using synthetic data for HSIC methods."""
        try:
            from prune_methods.depgraph_hsic import DepgraphHSICMethod
        except Exception:
            DepgraphHSICMethod = None

        if DepgraphHSICMethod is None or not isinstance(self.pruning_method, DepgraphHSICMethod):
            return 0

        if not hasattr(self.pruning_method, 'register_hooks'):
            return 0

        self.pruning_method.register_hooks()
        device = next(self.model.model.parameters()).device

        try:
            for i in range(num_samples):
                # Generate synthetic data
                synthetic_image = torch.randn(1, 3, 640, 640)
                synthetic_label = torch.randint(0, 80, (1,)).float()  # Convert to float for cdist compatibility

                # Forward pass
                with torch.no_grad():
                    self.model.model(synthetic_image.to(device))

                # Add labels for HSIC computation
                if hasattr(self.pruning_method, 'add_labels'):
                    self.pruning_method.add_labels(synthetic_label)

                # Cleanup
                del synthetic_image, synthetic_label
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        finally:
            self.pruning_method.remove_hooks()

        return num_samples

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
            
            # Call visualize_comparison if available
            if hasattr(self.pruning_method, 'visualize_comparison'):
                self.pruning_method.visualize_comparison()
            
            # Call visualize_pruned_filters if available
            if hasattr(self.pruning_method, 'visualize_pruned_filters'):
                self.pruning_method.visualize_pruned_filters()

    def save_pruning_results(self, path: str | Path) -> None:
        """Delegate result saving to the active pruning method."""
        if self.pruning_method is not None:
            self.logger.info("Saving pruning results to %s", path)
            self.pruning_method.save_results(path)

