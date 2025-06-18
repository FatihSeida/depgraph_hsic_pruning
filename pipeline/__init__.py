from importlib import import_module

__all__ = [
    "BasePruningPipeline",
    "PruningPipeline",
    "PipelineContext",
    "PipelineStep",
    "LoadModelStep",
    "TrainStep",
    "AnalyzeModelStep",
    "AnalyzeAfterTrainingStep",
    "GenerateMasksStep",
    "ApplyPruningStep",
    "ReconfigureModelStep",
    "CalcStatsStep",
    "ShortForwardPassStep",
    "CompareModelsStep",
    "MonitorComputationStep",
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
    step_modules = {
        "LoadModelStep": "load_model",
        "TrainStep": "train",
        "AnalyzeModelStep": "analyze",
        "AnalyzeAfterTrainingStep": "analyze",
        "GenerateMasksStep": "generate_masks",
        "ApplyPruningStep": "apply_pruning",
        "ReconfigureModelStep": "reconfigure",
        "CalcStatsStep": "calc_stats",
        "ShortForwardPassStep": "short_forward_pass",
        "CompareModelsStep": "compare",
        "MonitorComputationStep": "monitor_computation",
    }
    if name in step_modules:
        module = import_module(f"pipeline.step.{step_modules[name]}")
        return getattr(module, name)
    raise AttributeError(name)
