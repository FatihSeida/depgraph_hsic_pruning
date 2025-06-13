from dataclasses import dataclass

from helper.metric_manager import MetricManager


@dataclass
class DummyMetrics:
    mAP: float
    precision: float = 0.0
    other: int = 0


def test_record_training_with_dataclass():
    mgr = MetricManager()
    mgr.record_training(DummyMetrics(mAP=0.5, precision=0.7, other=1))
    assert mgr.training["mAP"] == 0.5
    assert mgr.training["precision"] == 0.7
    assert "other" not in mgr.training


def test_record_training_with_ultralytics_fields():
    mgr = MetricManager()
    metrics = {
        "metrics/precision": 0.1,
        "metrics/recall": 0.2,
        "metrics/mAP50": 0.3,
        "metrics/mAP50-95": 0.4,
    }
    mgr.record_training(metrics)
    assert mgr.training["precision"] == 0.1
    assert mgr.training["recall"] == 0.2
    assert mgr.training["mAP"] == 0.3
    assert mgr.training["mAP50_95"] == 0.4
