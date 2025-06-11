from pathlib import Path
from typing import Any, Dict
from abc import ABC, abstractmethod

from ultralytics_pruning import YOLO
from ultralytics_pruning.utils.torch_utils import get_flops, get_num_params


class BasePruningPipeline(ABC):
    """Abstract base class defining the pruning pipeline interface."""

    def __init__(self, model_path: str, data: str, workdir: str = "runs/pruning") -> None:
        self.model_path = model_path
        self.data = data
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.model: YOLO | None = None
        self.initial_stats: Dict[str, float] = {}
        self.pruned_stats: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}

    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        raise NotImplementedError

    @abstractmethod
    def calc_initial_stats(self) -> Dict[str, float]:
        """Calculate parameter count and FLOPs before pruning."""
        raise NotImplementedError

    @abstractmethod
    def pretrain(self, **train_kwargs: Any) -> Dict[str, Any]:
        """Optional pretraining step to run before pruning."""
        raise NotImplementedError

    @abstractmethod
    def analyze_structure(self) -> None:
        """Analyze model structure to guide pruning."""
        raise NotImplementedError

    @abstractmethod
    def generate_pruning_mask(self, ratio: float) -> None:
        """Generate pruning mask at ``ratio`` sparsity."""
        raise NotImplementedError

    @abstractmethod
    def apply_pruning(self) -> None:
        """Apply the previously generated pruning mask to the model."""
        raise NotImplementedError

    @abstractmethod
    def reconfigure_model(self) -> None:
        """Reconfigure the model after pruning if necessary."""
        raise NotImplementedError

    @abstractmethod
    def calc_pruned_stats(self) -> Dict[str, float]:
        """Calculate parameter count and FLOPs after pruning."""
        raise NotImplementedError

    @abstractmethod
    def finetune(self, **train_kwargs: Any) -> Dict[str, Any]:
        """Finetune the pruned model."""
        raise NotImplementedError

    def record_metrics(self) -> Dict[str, Any]:
        """Return a dictionary containing training and pruning statistics."""
        return {
            "initial": self.initial_stats,
            "pruned": self.pruned_stats,
            "training": self.metrics,
        }


class PruningPipeline(BasePruningPipeline):
    """Concrete implementation of :class:`BasePruningPipeline` for YOLO models."""

    def load_model(self) -> None:
        """Load the YOLO model from ``self.model_path``."""
        self.model = YOLO(self.model_path)

    def calc_initial_stats(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        params = get_num_params(self.model.model)
        flops = get_flops(self.model.model)
        self.initial_stats = {"parameters": params, "flops": flops}
        return self.initial_stats

    def pretrain(self, **train_kwargs: Any) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        metrics = self.model.train(data=self.data, **train_kwargs)
        self.metrics["pretrain"] = metrics
        return metrics or {}

    def analyze_structure(self) -> None:
        # Placeholder for user provided analysis logic
        pass

    def generate_pruning_mask(self, ratio: float) -> None:
        # Placeholder for mask generation logic
        pass

    def apply_pruning(self) -> None:
        # Placeholder for pruning application logic
        pass

    def reconfigure_model(self) -> None:
        # Optional step for layer reconfiguration
        pass

    def calc_pruned_stats(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        params = get_num_params(self.model.model)
        flops = get_flops(self.model.model)
        self.pruned_stats = {"parameters": params, "flops": flops}
        return self.pruned_stats

    def finetune(self, **train_kwargs: Any) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not loaded")
        metrics = self.model.train(data=self.data, **train_kwargs)
        self.metrics["finetune"] = metrics
        return metrics or {}

