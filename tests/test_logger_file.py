import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main


def test_log_file_created_with_logger(tmp_path, monkeypatch):
    class DummyPipeline:
        def __init__(self, *a, **k):
            self.model = types.SimpleNamespace(model=object())
            self.metrics_mgr = types.SimpleNamespace(to_csv=lambda p: Path(p))

        def load_model(self):
            pass

        def calc_initial_stats(self):
            pass

        def pretrain(self, **kw):
            pass

        def analyze_structure(self):
            pass

        def generate_pruning_mask(self, ratio):
            pass

        def apply_pruning(self):
            pass

        def reconfigure_model(self):
            pass

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

    monkeypatch.setattr(main, "PruningPipeline", DummyPipeline)
    logger = main.get_logger()
    cfg = main.TrainConfig(baseline_epochs=0, finetune_epochs=0, batch_size=1, ratios=[0])
    main.execute_pipeline("m", "d", None, 0, cfg, tmp_path, logger=logger)
    assert (tmp_path / "pipeline.log").exists()
