import os
import sys
import types
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.step.train import TrainStep
from pipeline.context import PipelineContext


class DummyTensor:
    def __init__(self, *shape):
        self.shape = shape
    def __len__(self):
        return self.shape[0]


def test_train_step_logs_warning_for_object_labels(monkeypatch, caplog):
    torch_stub = types.SimpleNamespace(
        tensor=lambda data: DummyTensor(len(data)),
        zeros=lambda *shape: DummyTensor(*shape),
        is_tensor=lambda t: isinstance(t, DummyTensor),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    ctx = PipelineContext(model_path="m", data="d")

    class DepgraphHSICMethod:
        def add_labels(self, labels):
            self.last = labels

    ctx.pruning_method = DepgraphHSICMethod()

    dummy_model = types.SimpleNamespace(callbacks={})

    def add_callback(event, cb):
        dummy_model.callbacks.setdefault(event, []).append(cb)

    def train(data=None, **kw):
        trainer = types.SimpleNamespace(
            batch={"img": torch_stub.zeros(1, 3, 1, 1), "cls": torch_stub.tensor([0])}
        )
        for cb in dummy_model.callbacks.get("on_train_batch_end", []):
            cb(trainer)
        return {}

    dummy_model.add_callback = add_callback
    dummy_model.train = train

    ctx.model = dummy_model

    step = TrainStep("phase", label_fn=lambda batch: torch_stub.tensor([1, 2]))

    with caplog.at_level(logging.WARNING):
        step.run(ctx)

    messages = [rec.message for rec in caplog.records]
    assert any("object-level" in m for m in messages)
