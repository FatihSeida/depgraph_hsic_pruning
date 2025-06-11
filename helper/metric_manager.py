from __future__ import annotations

"""Metric recording utilities for pruning experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


TRAINING_METRIC_FIELDS = [
    "mAP",
    "precision",
    "recall",
    "box_loss",
    "seg_loss",
    "objectness_loss",
    "cls_loss",
]

COMPUTATION_METRIC_FIELDS = [
    "elapsed_seconds",
    "total_time",
    "total_time_minutes",
    "gpu_utilization",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "ram_used_mb",
    "ram_total_mb",
    "ram_percent",
    "power_usage_watts",
]

PRUNING_METRIC_FIELDS = {
    "parameters": ["original", "pruned", "reduction", "reduction_percent"],
    "flops": ["original", "pruned", "reduction", "reduction_percent"],
    "filters": ["original", "pruned", "reduction", "reduction_percent"],
    "compression_ratio": None,
}


@dataclass
class MetricManager:
    """Accumulate training, computation and pruning metrics."""

    training: Dict[str, Any] = field(default_factory=dict)
    computation: Dict[str, Any] = field(default_factory=dict)
    pruning: Dict[str, Any] = field(default_factory=dict)

    def record_training(self, metrics: Dict[str, Any]) -> None:
        """Store training metrics filtered by :data:`TRAINING_METRIC_FIELDS`."""
        for field in TRAINING_METRIC_FIELDS:
            if field in metrics:
                self.training[field] = metrics[field]

    def record_computation(self, metrics: Dict[str, Any]) -> None:
        """Store computation metrics filtered by :data:`COMPUTATION_METRIC_FIELDS`."""
        for field in COMPUTATION_METRIC_FIELDS:
            if field in metrics:
                self.computation[field] = metrics[field]

    def record_pruning(self, metrics: Dict[str, Any]) -> None:
        """Store pruning metrics according to :data:`PRUNING_METRIC_FIELDS`."""
        for key, subfields in PRUNING_METRIC_FIELDS.items():
            if subfields is None and key in metrics:
                self.pruning[key] = metrics[key]
            elif key in metrics and isinstance(metrics[key], dict):
                self.pruning.setdefault(key, {})
                for sub in subfields or []:
                    if sub in metrics[key]:
                        self.pruning[key][sub] = metrics[key][sub]

    def as_dict(self) -> Dict[str, Any]:
        """Return all recorded metrics as a dictionary."""
        return {
            "training": self.training,
            "computation": self.computation,
            "pruning": self.pruning,
        }
