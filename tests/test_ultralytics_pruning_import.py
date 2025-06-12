import os
import sys
import types
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_ultralytics_pruning_lazy_import():
    # Stub models module with dummy YOLO class
    models = types.ModuleType("ultralytics_pruning.models")
    class YOLO:
        pass
    for name in ["NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"]:
        setattr(models, name, YOLO)
    sys.modules["ultralytics_pruning.models"] = models

    # Stub utilities required by __init__
    utils_mod = types.ModuleType("ultralytics_pruning.utils")
    utils_mod.ASSETS = {}
    utils_mod.SETTINGS = {}
    checks_mod = types.ModuleType("ultralytics_pruning.utils.checks")
    def check_yolo(*a, **k):
        pass
    checks_mod.check_yolo = check_yolo
    downloads_mod = types.ModuleType("ultralytics_pruning.utils.downloads")
    def download(*a, **k):
        pass
    downloads_mod.download = download
    utils_mod.checks = checks_mod
    utils_mod.downloads = downloads_mod
    sys.modules["ultralytics_pruning.utils"] = utils_mod
    sys.modules["ultralytics_pruning.utils.checks"] = checks_mod
    sys.modules["ultralytics_pruning.utils.downloads"] = downloads_mod

    # Stub pipeline module with dummy PruningPipeline
    pipeline_mod = types.ModuleType("pipeline")
    class PruningPipeline:
        pass
    pipeline_mod.PruningPipeline = PruningPipeline
    sys.modules["pipeline"] = pipeline_mod

    sys.modules.pop("ultralytics_pruning", None)
    module = importlib.import_module("ultralytics_pruning")

    try:
        assert module.YOLO is YOLO
        assert "PruningPipeline" not in module.__dict__
        assert module.PruningPipeline is PruningPipeline
    finally:
        for name in [
            "ultralytics_pruning",
            "ultralytics_pruning.models",
            "ultralytics_pruning.utils",
            "ultralytics_pruning.utils.checks",
            "ultralytics_pruning.utils.downloads",
            "pipeline",
        ]:
            sys.modules.pop(name, None)
