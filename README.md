# depgraph_hsic_pruning

This repository contains utilities based on the Ultralytics YOLO stack. A pruning
pipeline is provided for orchestrating model preparation, pruning and
fine-tuning.

## Installation

1. Install Python 3.8 or later.
2. Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

### Quick start

After installing the dependencies you can run the pruning pipeline as shown
below. The example expects the dataset to be defined in a YAML file following
Ultralytics' format with `train`, `val` and `nc` fields.

```python
from pipeline import PruningPipeline
from pipeline.step import (
    LoadModelStep,
    CalcStatsStep,
    TrainStep,
    AnalyzeModelStep,
    GenerateMasksStep,
    ApplyPruningStep,
)

steps = [
    LoadModelStep(),
    CalcStatsStep("initial"),
    TrainStep("pretrain", epochs=1),
    AnalyzeModelStep(),
    GenerateMasksStep(ratio=0.2),
    ApplyPruningStep(),
    CalcStatsStep("pruned"),
]

pipeline = PruningPipeline("yolov8n-seg.pt", data="coco8.yaml", steps=steps)
context = pipeline.run_pipeline()
print(context.metrics)
```

## Package structure

```
pipeline/
    base_pipeline.py   # abstract base class
    pruning_pipeline.py  # default implementation
    context.py          # shared pipeline state
    step/               # modular pipeline steps
ultralytics_pruning/     # fork of the Ultralytics package
```

Importing from `pipeline` exposes both `BasePruningPipeline` and
`PruningPipeline`.

## Extending `BasePruningPipeline`

To customize the pruning workflow create a subclass and implement the abstract
methods. The example below sketches out the required structure:

```python
from pipeline import BasePruningPipeline

class MyPipeline(BasePruningPipeline):
    def load_model(self):
        ...

    def calc_initial_stats(self):
        ...

    def pretrain(self, **kwargs):
        ...

    def analyze_structure(self):
        ...

    def generate_pruning_mask(self, ratio: float):
        ...

    def apply_pruning(self):
        ...

    def reconfigure_model(self):
        ...

    def calc_pruned_stats(self):
        ...

    def finetune(self, **kwargs):
        ...
```

Override each method with logic specific to your model architecture or pruning
strategy.

## Example usage

A complete workflow typically looks like the following:

```python
from pipeline import PruningPipeline
from pipeline.step import (
    LoadModelStep,
    CalcStatsStep,
    TrainStep,
    AnalyzeModelStep,
    GenerateMasksStep,
    ApplyPruningStep,
    ReconfigureModelStep,
)

steps = [
    LoadModelStep(),
    CalcStatsStep("initial"),
    TrainStep("pretrain", epochs=1),
    AnalyzeModelStep(),
    GenerateMasksStep(ratio=0.2),
    ApplyPruningStep(),
    ReconfigureModelStep(),
    CalcStatsStep("pruned"),
    TrainStep("finetune", epochs=3),
]

pipeline = PruningPipeline(
    model_path="yolov8n-seg.pt",
    data="coco8.yaml",  # dataset YAML with 'train', 'val' and 'nc'
    workdir="runs/pruning",
    steps=steps,
)

context = pipeline.run_pipeline()
print(pipeline.record_metrics())
```

The YAML file describing the dataset should have at least the following keys:

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 80  # number of classes
```

See the `ultralytics_pruning/cfg/datasets` directory for examples.

## Helper utilities

The ``helper`` package bundles small components used across the pruning
pipeline:

* ``ExperimentManager`` – records results for multiple pruning runs and
  visualises comparisons.
* ``MetricManager`` – collects training, computation and pruning metrics in a
  structured format.
* ``Logger`` – lightweight wrapper around :mod:`logging` used throughout the
  pipeline and pruning methods.

These utilities follow SOLID design principles to keep the codebase easy to
maintain.


## Batch training script

Use `main.py` to run all pruning methods across several ratios in one go:

```bash
python main.py --model yolov8n-seg.pt --data coco8.yaml \
    --baseline-epochs 1 --finetune-epochs 3 --batch-size 16 --ratios 0.2 0.4 0.6 0.8
```

Add `--resume` to continue interrupted runs.
