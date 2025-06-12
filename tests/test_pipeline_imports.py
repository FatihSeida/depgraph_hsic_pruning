import importlib
import types
import sys

# Stub heavy dependencies before importing pipeline modules
ultra_stub = types.SimpleNamespace(
    YOLO=object,
    utils=types.SimpleNamespace(
        torch_utils=types.SimpleNamespace(get_flops=lambda *a, **k: 0, get_num_params=lambda *a, **k: 0)
    ),
)
sys.modules.setdefault("ultralytics_pruning", ultra_stub)
sys.modules.setdefault("ultralytics_pruning.utils", ultra_stub.utils)
sys.modules.setdefault("ultralytics_pruning.utils.torch_utils", ultra_stub.utils.torch_utils)
torch_stub = types.SimpleNamespace(nn=types.SimpleNamespace(Conv2d=object))
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
matplotlib_stub = types.SimpleNamespace(pyplot=types.SimpleNamespace())
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", matplotlib_stub.pyplot)

from pipeline import (
    LoadModelStep,
    TrainStep,
    AnalyzeModelStep,
    GenerateMasksStep,
    ApplyPruningStep,
    ReconfigureModelStep,
    CalcStatsStep,
    CompareModelsStep,
)


def test_step_classes_resolve_from_pipeline():
    assert LoadModelStep is importlib.import_module("pipeline.step.load_model").LoadModelStep
    assert TrainStep is importlib.import_module("pipeline.step.train").TrainStep
    assert AnalyzeModelStep is importlib.import_module("pipeline.step.analyze").AnalyzeModelStep
    assert GenerateMasksStep is importlib.import_module("pipeline.step.generate_masks").GenerateMasksStep
    assert ApplyPruningStep is importlib.import_module("pipeline.step.apply_pruning").ApplyPruningStep
    assert ReconfigureModelStep is importlib.import_module("pipeline.step.reconfigure").ReconfigureModelStep
    assert CalcStatsStep is importlib.import_module("pipeline.step.calc_stats").CalcStatsStep
    assert CompareModelsStep is importlib.import_module("pipeline.step.compare").CompareModelsStep
