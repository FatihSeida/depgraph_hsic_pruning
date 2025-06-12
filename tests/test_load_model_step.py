from unittest.mock import MagicMock
import sys
import types
import os

# Stub plotting library required by helper utilities
matplotlib_stub = types.SimpleNamespace(pyplot=types.SimpleNamespace())
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", matplotlib_stub.pyplot)
torch_stub = types.SimpleNamespace(nn=types.SimpleNamespace())
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.context import PipelineContext

dummy = MagicMock(name="YOLO")
module = types.SimpleNamespace(YOLO=MagicMock(return_value=dummy))
sys.modules.setdefault("ultralytics_pruning", module)
from pipeline.step.load_model import LoadModelStep


def test_load_model_step_updates_context():
    ctx = PipelineContext(model_path="yolov8n.yaml", data="data.yaml")
    step = LoadModelStep()
    step.run(ctx)
    module.YOLO.assert_called_with("yolov8n.yaml")
    assert ctx.model is dummy
