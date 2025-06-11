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

pipeline = PruningPipeline("yolov8n-seg.pt", data="coco8.yaml")
pipeline.load_model()
pipeline.calc_initial_stats()
pipeline.pretrain(epochs=1)
pipeline.analyze_structure()
pipeline.generate_pruning_mask(ratio=0.2)
pipeline.apply_pruning()
pipeline.calc_pruned_stats()
pipeline.finetune(epochs=3)
print(pipeline.record_metrics())
```

## Package structure

```
pipeline/
    base_pipeline.py   # abstract base class
    pruning_pipeline.py  # default implementation
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

pipeline = PruningPipeline(
    model_path="yolov8n-seg.pt",
    data="coco8.yaml",  # dataset YAML with 'train', 'val' and 'nc'
    workdir="runs/pruning"
)

pipeline.load_model()
initial = pipeline.calc_initial_stats()
print(f"Initial stats: {initial}")

pipeline.pretrain(epochs=1)

pipeline.analyze_structure()
pipeline.generate_pruning_mask(ratio=0.2)

pipeline.apply_pruning()
pipeline.reconfigure_model()
pruned = pipeline.calc_pruned_stats()
print(f"After pruning: {pruned}")

pipeline.finetune(epochs=3)
print(pipeline.record_metrics())
```

The YAML file describing the dataset should have at least the following keys:

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 80  # number of classes
```

See the `ultralytics_pruning/cfg/datasets` directory for examples.

