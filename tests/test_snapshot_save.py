import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main

class DummyPipeline:
    def __init__(self, *a, **k):
        self.workdir = Path(k.get("workdir", ""))
        self.model = types.SimpleNamespace(
            model=object(),
            save=lambda p: Path(p).write_text("model")
        )
        self.pruning_method = None
        self.metrics_mgr = types.SimpleNamespace(
            to_csv=lambda p: Path(p),
            record_computation=lambda m: None,
        )
    def load_model(self):
        pass
    def calc_initial_stats(self):
        pass
    def pretrain(self, **kw):
        pass
    def analyze_structure(self):
        pass
    def set_pruning_method(self, method):
        self.pruning_method = method
    def generate_pruning_mask(self, ratio):
        pass
    def apply_pruning(self):
        pass
    def reconfigure_model(self, output_path=None):
        self.output_path = output_path
    def calc_pruned_stats(self):
        pass
    def finetune(self, **kw):
        pass
    def visualize_results(self):
        pass
    def save_pruning_results(self, path):
        pass
    def save_metrics_csv(self, path):
        return Path(path)
    def save_model(self, path):
        Path(path).write_text("snap")
        self.saved = Path(path)

class DummyMethod:
    def __init__(self, model=None, workdir=None):
        self.model = model


def test_snapshot_and_pruned_model_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PruningPipeline", DummyPipeline)
    monkeypatch.setattr(main, "YOLO", lambda p: types.SimpleNamespace(model=object(), save=lambda x: None))
    cfg = main.TrainConfig(baseline_epochs=0, finetune_epochs=0, batch_size=1, ratios=[0.2])
    pipeline, _ = main.execute_pipeline("m", "d", DummyMethod, 0.2, cfg, tmp_path)
    assert (tmp_path / "snapshot.pt").exists()
    assert pipeline.saved == tmp_path / "snapshot.pt"
    assert pipeline.output_path == tmp_path / "pruned_model_DummyMethod_0.2.pt"
    assert pipeline.pruning_method.model is pipeline.model.model

