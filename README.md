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
The requirements file now includes the `ultralytics` package (YOLOv8) so it will
be installed automatically.

### Quick start

After installing the dependencies you can run the pruning pipeline as shown
below. The example expects the dataset to be defined in a YAML file following
Ultralytics' format with `train`, `val` and `nc` fields.

```python
from pipeline import PruningPipeline
from prune_methods import DepgraphHSICMethod
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
    TrainStep("pretrain", epochs=1, plots=True),
    AnalyzeModelStep(),
    GenerateMasksStep(ratio=0.2),
    ApplyPruningStep(),
    CalcStatsStep("pruned"),
]

pipeline = PruningPipeline(
    "yolov8n-seg.pt",
    data="biotech_model_train.yaml",
    pruning_method=DepgraphHSICMethod(None),  # model assigned by LoadModelStep
    steps=steps,
)
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
ultralytics/     # fork of the Ultralytics package
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
    TrainStep("pretrain", epochs=1, plots=True),
    AnalyzeModelStep(),
    GenerateMasksStep(ratio=0.2),
    ApplyPruningStep(),
    ReconfigureModelStep(),
    CalcStatsStep("pruned"),
    TrainStep("finetune", epochs=3, plots=True),
]

pipeline = PruningPipeline(
    model_path="yolov8n-seg.pt",
    data="biotech_model_train.yaml",  # dataset YAML with 'train', 'val' and 'nc'
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

See the `ultralytics/cfg/datasets` directory for examples.

## Helper utilities

The ``helper`` package bundles small components used across the pruning
pipeline:

* ``ExperimentManager`` – records results for multiple pruning runs and
  visualises comparisons.
* ``MetricManager`` – collects training, computation and pruning metrics in a
  structured format.
* ``Logger`` – lightweight wrapper around :mod:`logging` used throughout the
  pipeline and pruning methods.
* ``plot_metric_heatmaps`` – create heatmaps of metrics grouped by pruning
  method and ratio.

These utilities follow SOLID design principles to keep the codebase easy to
maintain.

Example of generating heatmaps for pruning results:

```python
import pandas as pd
from helper import plot_metric_heatmaps

df = pd.DataFrame({
    "pruning.method": ["l1", "l1", "l2", "l2"],
    "pruning.ratio": [0.2, 0.4, 0.2, 0.4],
    "FLOPsReduction": [0.1, 0.2, 0.3, 0.4],
})
plot_metric_heatmaps(df, ["FLOPsReduction"], "plots")
```

Metrics recorded by experiment runs can also be visualised directly. Use
``load_metrics_dataframe`` to read all ``metrics.csv`` files from a directory and
then create heatmaps for common metrics:

```python
from helper import load_metrics_dataframe, plot_metric_heatmaps, DEFAULT_METRICS

df = load_metrics_dataframe("runs/experiments")
plot_metric_heatmaps(df, DEFAULT_METRICS, "plots")
```


## Batch training script

Use `main.py` to run all pruning methods across several ratios in one go:

```bash
python main.py --model yolov8n-seg.pt \
    --baseline-epochs 1 --finetune-epochs 3 --batch-size 16 --ratios 0.2 0.4 0.6 0.8
```
The dataset defaults to `biotech_model_train.yaml` if `--data` is not supplied.
Available values for `--methods`:

* `l1`
* `random`
* `depgraph`
* `depgraph_hsic`
* `tp_random`
* `isomorphic`
* `hsic_lasso`
* `whc`

The `depgraph_hsic` option maps to the ``DepgraphHSICMethod`` class. It collects
feature activations and labels during forward passes, calculates HSIC scores to
measure channel dependence on the targets and ranks them using ``LassoLars``.
Pruning is then applied through ``torch-pruning``'s ``DependencyGraph`` to keep
tensor shapes consistent.

Add `--resume` to continue interrupted runs.
Add `--heatmap-only` to generate heatmap visualizations without line plots.
If baseline weights are already present in the working directory they will be
reused by default, which skips the initial pretraining step. Disable this by
setting ``reuse_baseline=False`` in ``TrainConfig`` if fresh baseline training
is required.

Each run directory will contain a `pipeline.log` file capturing detailed
training and pruning output for the selected ratio.

Use `--device` to select the training device (defaults to `cuda:0`).
The script automatically visualizes several metrics after all runs:
`FLOPsReduction`, `FilterReduction`, `TotalRuntimeMinutes`, `Recall`,
`mAP50_95`, `ParameterReduction`, `Precision`, `mAP`, `ModelSizeMB` and
`AvgMemoryUsageMB`.

## Panduan Setup Lingkungan (Bahasa Indonesia)

Apabila Anda menggunakan server bersama dan tidak dapat menginstal paket secara
global, buatlah *environment* terpisah dengan `conda`:

```bash
conda create -n depgraph-env python=3.8
conda activate depgraph-env
```

Setelah environment aktif, jalankan perintah berikut untuk menginstal semua
dependensi yang tercantum di `requirements.txt`:

```bash
pip install -r requirements.txt
```

File `requirements.txt` sudah mencakup library penting seperti
`ultralytics` dan `ultralytics-thop` yang menyediakan modul `thop`.
Pastikan perintah di atas dijalankan di dalam environment yang telah
diaktifkan agar seluruh fitur proyek dapat berfungsi dengan baik.
