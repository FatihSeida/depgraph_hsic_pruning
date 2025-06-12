from .context import PipelineContext
from .step import PipelineStep

# Step classes are imported lazily by consumers to avoid heavy dependencies at
# package import time. They are re-exported here for convenience.
from importlib import import_module


def __getattr__(name: str):
    if name in {
        "LoadModelStep",
        "TrainStep",
        "AnalyzeModelStep",
        "GenerateMasksStep",
        "ApplyPruningStep",
        "ReconfigureModelStep",
        "CalcStatsStep",
        "CompareModelsStep",
    }:
        module = import_module(f"pruning_pipeline.step.{name.lower()}")
        return getattr(module, name)
    raise AttributeError(name)

__all__ = [
    "PipelineContext",
    "PipelineStep",
    "LoadModelStep",
    "TrainStep",
    "AnalyzeModelStep",
    "GenerateMasksStep",
    "ApplyPruningStep",
    "ReconfigureModelStep",
    "CalcStatsStep",
    "CompareModelsStep",
]
