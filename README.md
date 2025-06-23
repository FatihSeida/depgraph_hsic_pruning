# Depgraph HSIC Pruning

This repository provides utilities and pruning methods for YOLO models.

## Installation

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

For running the test suite use:

```bash
pip install -r requirements-test.txt
```

### DepgraphHSICMethod

`DepgraphHSICMethod` requires `scikit-learn` for its HSIC-Lasso implementation.
The package is listed in `requirements.txt`, but if you only installed a subset of
dependencies make sure `scikit-learn` is available:

```bash
pip install scikit-learn
```

### FLOP statistics

FLOP statistics are computed using a built-in manual method. Earlier versions
relied on the optional `ultralytics-thop` package, but this dependency has been
removed. The manual calculation works without extra packages, though the result
may be slightly less accurate than dedicated profiling tools.

## Usage

Run `main.py` to train and prune a YOLO model. Specify the computation device
with the `--device` option. If the loaded model supports `.to()`, it will be
moved to that device automatically.

When using `DepgraphHSICMethod`, ensure that `analyze_structure()` is run after every training phase. The `execute_pipeline` helper automatically performs this analysis just before pruning mask generation.

`DepgraphHSICMethod.apply_pruning` relies on `torch_pruning`'s
`DependencyGraph` to remove pruned channels and handle layer dependencies
automatically. If layers are replaced or reconstructed during training,
the method rebuilds the graph automatically, so you generally do not
need to manually adjust the model before pruning. Always call
`analyze_model()` before applying the pruning plan or invoke
``apply_pruning(rebuild=True)`` to rebuild the graph automatically.
Should pruning fail because a layer is reported as missing from the
dependency graph, rerun ``apply_pruning(rebuild=True)`` to rebuild the
graph before applying the mask. Avoid manually replacing layers unless
you fully understand the dependency structure.

Example usage:

```python
method.generate_pruning_mask(0.5)
method.apply_pruning()
```

When calling ``generate_pruning_mask`` you may optionally provide a
``DataLoader``. The method will perform a short inference run over the
dataloader, recording activations through its registered hooks and storing the
labels via :func:`add_labels`.  This can be useful when pruning without running
the training pipeline:

```python
method.analyze_model()
method.generate_pruning_mask(0.5, dataloader=loader)
```

HSIC pruning relies on comparing activations across multiple label batches.
If fewer than ``min_labels`` batches are recorded (``min_labels`` defaults to
``4``), ``generate_pruning_mask`` raises a ``ValueError``.

When no activations or labels were recorded, the method raises a
``RuntimeError`` instead of falling back to a simple L1-norm based plan.

``ShortForwardPassStep`` only collects two labelled images.  When relying on
this step alone, call ``generate_pruning_mask`` with ``min_labels=2`` or run
additional passes to gather more labels.

