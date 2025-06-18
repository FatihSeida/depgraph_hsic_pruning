import importlib
import pytest

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "torch",
    "sklearn",
    "matplotlib",
    "seaborn",
    "torch_pruning",
]

missing = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]

if missing:
    pytest.skip(
        "Missing test dependencies: {}. Install them with 'pip install -r requirements-test.txt'".format(
            ", ".join(missing)
        ),
        allow_module_level=True,
    )
