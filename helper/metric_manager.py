from __future__ import annotations

"""Metric recording utilities for pruning experiments."""

from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import csv


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
    "avg_ram_used_mb",
    "power_usage_watts",
]

PRUNING_METRIC_FIELDS = {
    "parameters": ["original", "pruned", "reduction", "reduction_percent"],
    "flops": ["original", "pruned", "reduction", "reduction_percent"],
    "filters": ["original", "pruned", "reduction", "reduction_percent"],
    "model_size_mb": ["original", "pruned", "reduction", "reduction_percent"],
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
        if not isinstance(metrics, dict):
            if is_dataclass(metrics):
                metrics = asdict(metrics)
            elif hasattr(metrics, "__dict__"):
                metrics = metrics.__dict__
            else:
                metrics = {}
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

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        """Write recorded metrics to ``path`` as a single-row CSV."""

        def _flatten(d: Dict[str, Any], prefix: str = "", out: Dict[str, Any] | None = None) -> Dict[str, Any]:
            if out is None:
                out = {}
            for key, val in d.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(val, dict):
                    _flatten(val, name, out)
                else:
                    out[name] = val
            return out

        flat = _flatten(self.as_dict())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat))
            writer.writeheader()
            writer.writerow({k: flat.get(k, "") for k in sorted(flat)})
        return path
