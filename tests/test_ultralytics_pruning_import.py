import os
import sys
import types
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_ultralytics_lazy_import():
    # Stub models module with dummy YOLO class
    models = types.ModuleType("ultralytics.models")
    class YOLO:
        pass
    for name in ["NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"]:
        setattr(models, name, YOLO)
    sys.modules["ultralytics.models"] = models

    # Stub utilities required by __init__
    utils_mod = types.ModuleType("ultralytics.utils")
    utils_mod.ASSETS = {}
    utils_mod.SETTINGS = {}
    checks_mod = types.ModuleType("ultralytics.utils.checks")
    def check_yolo(*a, **k):
        pass
    checks_mod.check_yolo = check_yolo
    downloads_mod = types.ModuleType("ultralytics.utils.downloads")
    def download(*a, **k):
        pass
    downloads_mod.download = download
    utils_mod.checks = checks_mod
    utils_mod.downloads = downloads_mod
    sys.modules["ultralytics.utils"] = utils_mod
    sys.modules["ultralytics.utils.checks"] = checks_mod
    sys.modules["ultralytics.utils.downloads"] = downloads_mod

    # Stub pipeline module with dummy PruningPipeline
    pipeline_mod = types.ModuleType("pipeline")
    class PruningPipeline:
        pass
    pipeline_mod.PruningPipeline = PruningPipeline
    sys.modules["pipeline"] = pipeline_mod

    root_mod = types.ModuleType("ultralytics")

    def __getattr__(name):
        if name == "YOLO":
            return YOLO
        if name == "PruningPipeline":
            return PruningPipeline
        raise AttributeError(name)

    root_mod.__getattr__ = __getattr__
    sys.modules.pop("ultralytics", None)
    sys.modules["ultralytics"] = root_mod
    module = importlib.import_module("ultralytics")

    try:
        assert module.YOLO is YOLO
        assert "PruningPipeline" not in module.__dict__
        assert module.PruningPipeline is PruningPipeline
    finally:
        for name in [
            "ultralytics",
            "ultralytics.models",
            "ultralytics.utils",
            "ultralytics.utils.checks",
            "ultralytics.utils.downloads",
            "pipeline",
        ]:
            sys.modules.pop(name, None)
