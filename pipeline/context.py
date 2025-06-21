from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from helper import Logger, MetricManager, get_logger
from prune_methods.base import BasePruningMethod


@dataclass
class PipelineContext:
    """Container for objects shared across pipeline steps."""

    model_path: str
    data: str
    workdir: Path = Path("runs/pruning")
    pruning_method: Optional[BasePruningMethod] = None
    logger: Logger = field(default_factory=get_logger)
    model: Any = None
    dataloader: Any = None
    initial_stats: Dict[str, float] = field(default_factory=dict)
    pruned_stats: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_mgr: MetricManager = field(init=False)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.metrics_mgr = MetricManager()

__all__ = ["PipelineContext"]
