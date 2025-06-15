# depgraph_hsic_pruning

This repository contains utilities based on the Ultralytics YOLO stack.
A pruning pipeline is provided for orchestrating model preparation, pruning and fine-tuning.

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
Ultralytics' format with `train`, `val` and `nc` fields. When using
`DepgraphHSICMethod` you must record labels for every forward pass and run a
training or validation step before analyzing the model so activations and
labels are available.

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

Example log output:

```
INFO - Calculating pruned statistics
metric         initial  pruned  reduction  %
parameters     27000    18000   9000       33.3%
flops          100.0    70.0    30.0       30.0%
filters        100      80      20         20.0%
model_size_mb  4.5      3.0     1.5       33.3%
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

Add `--resume` to continue interrupted runs. If `weights/last.pt` is missing or
`weights/best.pt` exists with `results.csv` showing that all configured epochs
completed, training starts from scratch instead of resuming and a message is
logged.
Add `--heatmap-only` to generate heatmap visualizations without line plots.
If baseline weights are already present in the working directory they will be
reused by default, which skips the initial pretraining step. Disable this by
setting ``reuse_baseline=False`` in ``TrainConfig`` if fresh baseline training
is required.

``DepgraphHSICMethod`` needs activations and labels obtained from a forward
pass to compute pruning scores. When ``reuse_baseline=True`` the initial
training can be skipped, but you should still run a training or evaluation step
before ``AnalyzeModelStep`` so that activations and labels are populated for
scoring. When using :class:`~pipeline.step.train.TrainStep` labels are
automatically recorded after each batch. For custom loops this must be done
manually. Call ``DepgraphHSICMethod.add_labels()`` immediately after
``model(images)`` for every batch and before any new forward pass occurs;
otherwise ``generate_pruning_mask`` will raise a mismatch error:

```python
for images, labels in dataloader:
    outputs = model(images)
    pruning_method.add_labels(labels)
    loss = loss_fn(outputs, labels)
    ...
```

``DepgraphHSICMethod`` expects labels to correspond one-to-one with each
forward pass. YOLO detection batches supply one label per object, so passing
``batch["cls"]`` directly will produce a mismatch error. Aggregate them
before calling ``add_labels``—for example by computing a single class index for
each image:

```python
cls = batch["cls"].view(batch["img"].shape[0], -1)
image_labels = cls[:, 0]
pruning_method.add_labels(image_labels)
```

The same aggregation can be automated when using
``TrainStep`` by providing a ``label_fn``:

```python
def aggregate_labels(batch):
    cls = batch["cls"].view(batch["img"].shape[0], -1)
    return cls[:, 0]

steps = [
    TrainStep("pretrain", label_fn=aggregate_labels, epochs=1),
    ...
]
```

``DepgraphHSICMethod`` will call ``label_fn`` for each batch before adding
labels internally.

Calling ``analyze_model()`` clears any stored activations, recorded layer
shapes, activation counters and labels so that subsequent training phases
start with a clean slate.

Automatic recording for training pipelines is implemented in
``pipeline/step/train.py`` where :class:`~pipeline.step.train.TrainStep` adds a
callback on ``on_train_batch_end`` to store ``batch["cls"]`` after each forward
pass.

Each run directory will contain a `pipeline.log` file capturing detailed
training and pruning output for the selected ratio.
Enabling `--debug` will log a message whenever batch labels are recorded,
including their shape.

Use `--device` to select the training device (defaults to `cuda:0`).
The script automatically visualizes several metrics after all runs:
`FLOPsReduction`, `FilterReduction`, `TotalRuntimeMinutes`, `Recall`,
`mAP50_95`, `ParameterReduction`, `Precision`, `mAP`, `ModelSizeMB` and
`AvgMemoryUsageMB`.

## Running tests

The test suite relies on additional packages not included in the pruning
pipeline requirements. **Install `pandas`, `numpy`, and `torch` before running
`pytest`.** You can install them directly or use the provided convenience file
`requirements-test.txt`:

```bash
pip install -r requirements-test.txt
```

Alternatively install them manually:

```bash
pip install torch pandas numpy
```

Once the dependencies are available, run the tests with:

```bash
pytest
```

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
