import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub heavy dependencies so importing main works
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch.nn'] = types.ModuleType('torch.nn')
up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_num_params = lambda *a, **k: 0
utils.torch_utils = torch_utils
up.utils = utils
up.YOLO = lambda *a, **k: None
sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils
mpl = types.ModuleType('matplotlib')
plt = types.ModuleType('matplotlib.pyplot')
sys.modules['matplotlib'] = mpl
sys.modules['matplotlib.pyplot'] = plt

import main

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
    def generate_pruning_mask(self, ratio, dataloader=None):
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
    def record_metrics(self):
        return {}

class DummyMgr:
    def __init__(self, *a, **k):
        pass
    def add_result(self, *a, **k):
        pass
    def compare_pruning_methods(self):
        pass
    def plot_line(self, *a, **k):
        pass
    def plot_heatmap(self, *a, **k):
        pass


def test_baseline_reused_when_available(monkeypatch, tmp_path):
    calls = []
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)

    def fake_exec(model, data, method_cls, ratio, cfg, workdir, **kw):
        calls.append(str(model))
        return DummyPipeline(), workdir / "metrics.csv"

    monkeypatch.setattr(main, "execute_pipeline", fake_exec)
    monkeypatch.setattr(main, "ExperimentManager", DummyMgr)

    weights_dir = tmp_path / "baseline" / "baseline" / "weights"
    weights_dir.mkdir(parents=True)
    weight_file = weights_dir / "best.pt"
    weight_file.write_text("w")

    cfg = main.TrainConfig(baseline_epochs=1, finetune_epochs=1, batch_size=1, ratios=[0.2])
    runner = main.ExperimentRunner("m.pt", "d.yaml", [type("M", (), {})], cfg, workdir=tmp_path)
    runner.run()

    assert calls == [str(weight_file)]
