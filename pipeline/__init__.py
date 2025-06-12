from importlib import import_module
import re


def camel_to_snake(name: str) -> str:
    """Convert CamelCase names to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()

__all__ = [
    "BasePruningPipeline",
    "PruningPipeline",
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


def __getattr__(name: str):
    if name == "BasePruningPipeline":
        return import_module("pipeline.base_pipeline").BasePruningPipeline
    if name == "PruningPipeline":
        return import_module("pipeline.pruning_pipeline").PruningPipeline
    if name == "PipelineContext":
        return import_module("pipeline.context").PipelineContext
    if name == "PipelineStep":
        return import_module("pipeline.step").PipelineStep
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
        module_name = camel_to_snake(name)
        if module_name.endswith("_step"):
            module_name = module_name[:-5]
        try:
            module = import_module(f"pipeline.step.{module_name}")
        except ModuleNotFoundError:
            if module_name.endswith("_model"):
                module_name = module_name[:-6]
            elif module_name.endswith("_models"):
                module_name = module_name[:-7]
            module = import_module(f"pipeline.step.{module_name}")
        return getattr(module, name)
    raise AttributeError(name)
