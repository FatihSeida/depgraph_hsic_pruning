from unittest.mock import MagicMock
import sys
import types
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a lightweight stub for prune_methods.base so importing PipelineContext
# does not require heavy dependencies like torch during tests.
if "prune_methods.base" not in sys.modules:
    stub = types.ModuleType("prune_methods.base")

    class BasePruningMethod:  # pragma: no cover - simple placeholder
        pass

    stub.BasePruningMethod = BasePruningMethod
    sys.modules["prune_methods.base"] = stub

from pipeline.context import PipelineContext

dummy = MagicMock(name="YOLO")
module = types.SimpleNamespace(YOLO=MagicMock(return_value=dummy))
sys.modules["ultralytics_pruning"] = module
from pipeline.step.load_model import LoadModelStep


def test_load_model_step_updates_context():
    ctx = PipelineContext(model_path="yolov8n.yaml", data="data.yaml")
    step = LoadModelStep()
    step.run(ctx)
    module.YOLO.assert_called_with("yolov8n.yaml")
    assert ctx.model is dummy
