from __future__ import annotations

import time
from typing import Dict, Any

from helper.metric_manager import MetricManager
from helper.metrics.computation_metrics import GPUMetric, MemoryMetric, PowerMetric

from ..context import PipelineContext
from . import PipelineStep


class MonitorComputationStep(PipelineStep):
    """Sample GPU, memory and power usage around a training phase."""

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.gpu = GPUMetric()
        self.memory = MemoryMetric()
        self.power = PowerMetric()
        self._start_time = 0.0
        self._start_ram_used_mb = 0.0

    def start(self) -> None:
        """Begin metric collection."""
        self._start_time = time.time()
        self.gpu.reset()
        self.memory.reset()
        self.power.reset()
        self.gpu.collect()
        mem = self.memory.collect()
        self._start_ram_used_mb = mem.get("ram_used_mb", 0.0)
        self.power.collect()

    def stop(self, mgr: MetricManager) -> Dict[str, Any]:
        """Stop metric collection and record summary using ``mgr``."""
        elapsed = time.time() - self._start_time
        gpu_metrics = self.gpu.collect()
        mem_metrics = self.memory.collect()
        avg_ram_used = (self._start_ram_used_mb + mem_metrics.get("ram_used_mb", 0.0)) / 2.0
        self.power.collect(interval=elapsed)
        gpu_summary = self.gpu.get_summary()
        mem_summary = self.memory.get_summary()
        power_summary = self.power.get_summary()
        metrics = {
            "elapsed_seconds": elapsed,
            "total_time": elapsed,
            "total_time_minutes": elapsed / 60.0,
            "gpu_utilization": gpu_summary.get("avg_utilization", 0.0),
            "gpu_memory_percent": gpu_summary.get("avg_memory_percent", 0.0),
            "gpu_memory_used_mb": gpu_metrics.get("gpu_memory_used_mb", 0.0),
            "gpu_memory_total_mb": gpu_metrics.get("gpu_memory_total_mb", 0.0),
            "ram_used_mb": mem_metrics.get("ram_used_mb", 0.0),
            "ram_total_mb": mem_metrics.get("ram_total_mb", 0.0),
            "ram_percent": mem_summary.get("avg_ram_percent", 0.0),
            "avg_ram_used_mb": avg_ram_used,
            "power_usage_watts": power_summary.get("avg_power_watts", 0.0),
        }
        mgr.record_computation(metrics)
        return metrics

    def run(self, context: PipelineContext) -> None:  # pragma: no cover - not used in tests
        metrics = self.stop(context.metrics_mgr)
        context.metrics.setdefault("computation", {})[self.phase] = metrics



__all__ = ["MonitorComputationStep"]
