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
