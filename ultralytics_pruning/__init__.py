# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.99"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics_pruning.models import NAS, RTDETR, SAM, YOLO, YOLOE, FastSAM, YOLOWorld
from importlib import import_module
from ultralytics_pruning.utils import ASSETS, SETTINGS
from ultralytics_pruning.utils.checks import check_yolo as checks
from ultralytics_pruning.utils.downloads import download


def __getattr__(name: str):
    if name == "PruningPipeline":
        return import_module("pipeline").PruningPipeline
    raise AttributeError(name)

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "PruningPipeline",
    "checks",
    "download",
    "settings",
)
