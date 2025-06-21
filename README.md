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

## Usage

Run `main.py` to train and prune a YOLO model. Specify the computation device
with the `--device` option. If the loaded model supports `.to()`, it will be
moved to that device automatically.

When using `DepgraphHSICMethod`, ensure that `analyze_structure()` is run after every training phase. The `execute_pipeline` helper automatically performs this analysis just before pruning mask generation.

`DepgraphHSICMethod.apply_pruning` relies on `torch_pruning`'s
`DependencyGraph` to remove pruned channels. If layers are replaced or
reconstructed during training, the method rebuilds the graph
automatically, so you generally do not need to manually adjust the
model before pruning.

Example usage:

```python
method.generate_pruning_mask(0.5, dataloader=train_loader)
method.apply_pruning()
```
