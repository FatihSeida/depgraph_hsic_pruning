# Pipeline Module Overview

This directory implements a modular pruning workflow used throughout the project. The design follows a step-based approach so that individual actions (loading a model, training, pruning, etc.) can be composed and reused.

## File layout

- **`base_pipeline.py`** – Defines `BasePruningPipeline`, an abstract class listing all operations required for a pruning run.
- **`pruning_pipeline.py`** – Default implementation built around Ultralytics' YOLO models. It accepts a list of steps and executes them in order via `run_pipeline()`.
- **`context.py`** – Declares `PipelineContext`, a dataclass that holds objects shared across steps such as the model instance, statistics and training metrics.
- **`model_reconfig.py`** – Provides utilities for adjusting layer shapes after pruning. `AdaptiveLayerReconfiguration` is used by `PruningPipeline` when `ReconfigureModelStep` is executed.
- **`step/`** – Contains concrete `PipelineStep` subclasses that perform small actions. Examples include `LoadModelStep`, `TrainStep`, `MonitorComputationStep`, `AnalyzeModelStep` and more.

## PipelineContext

`PipelineContext` is created at the beginning of `PruningPipeline.run_pipeline()` and passed to each step. It stores values like `model_path`, `data`, a logging object and mutable fields such as `model`, `initial_stats` and `metrics`.

```python
@dataclass
class PipelineContext:
    model_path: str
    data: str
    workdir: Path = Path("runs/pruning")
    pruning_method: Optional[BasePruningMethod] = None
    logger: Logger = field(default_factory=get_logger)
    model: Any = None
    initial_stats: Dict[str, float] = field(default_factory=dict)
    pruned_stats: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
```

*(see `pipeline/context.py`)*

## Steps and execution flow

Each step subclass implements `run(context)` and mutates the provided `PipelineContext` in place. Example implementation:

```python
class LoadModelStep(PipelineStep):
    def run(self, context: PipelineContext) -> None:
        context.logger.info("Loading model from %s", context.model_path)
        context.model = YOLO(context.model_path)
```

*(see `pipeline/step/load_model.py`)*

`PruningPipeline.run_pipeline()` iterates over the configured steps:

```python
for step in self.steps:
    step.run(context)
```

After all steps have run, results stored in the context are copied back to the pipeline instance so they can be accessed directly from the pipeline object.

## Typical workflow

A typical set of steps might look as follows:

1. `LoadModelStep()`
2. `CalcStatsStep("initial")`
3. `AnalyzeModelStep()`
4. `MonitorComputationStep("pretrain")` *(start before training)*
5. `TrainStep("pretrain", epochs=1, plots=True)`
6. `MonitorComputationStep("pretrain")` *(stop after training)*
7. `GenerateMasksStep(ratio=0.2)`
8. `ApplyPruningStep()`
9. `ReconfigureModelStep()`
10. `CalcStatsStep("pruned")`
11. `MonitorComputationStep("finetune")` *(start before training)*
12. `TrainStep("finetune", epochs=3, plots=True)`
13. `MonitorComputationStep("finetune")` *(stop after training)*

`AnalyzeModelStep` registers forward hooks and clears previously recorded
activations or statistics, so a training pass must follow it to populate fresh
data for pruning.

`PruningPipeline` will execute them sequentially, passing the same context object to each. Statistics and metrics are accumulated inside `context` and can be retrieved at the end via `pipeline.record_metrics()` or directly from `context.metrics`.

This modular structure makes it easy to customise or extend the pruning process by defining new step classes or reordering existing ones.

### Using ``DepgraphHSICMethod``

Any pruning algorithm subclassing ``BasePruningMethod`` can be attached to the pipeline. ``DepgraphHSICMethod`` ranks channels via HSIC scores and prunes them through a dependency graph:

```python
from prune_methods import DepgraphHSICMethod

pipeline = PruningPipeline(
    "yolov8n-seg.pt",
    data="dataset.yaml",
    pruning_method=DepgraphHSICMethod(model),
    steps=steps,
)
```
When the built-in ``TrainStep`` is used labels are captured automatically after
each batch. If you train the model manually remember to call
``DepgraphHSICMethod.add_labels`` so that activations and targets stay aligned.

