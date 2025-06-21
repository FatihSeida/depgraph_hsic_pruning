import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib


def test_execute_pipeline_moves_model_to_device(monkeypatch, tmp_path):
    # stub heavy deps
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')

    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_flops = lambda *a, **k: 0
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    up.utils = utils
    sys.modules['ultralytics'] = up
    sys.modules['ultralytics.utils'] = utils
    sys.modules['ultralytics.utils.torch_utils'] = torch_utils

    mpl = types.ModuleType('matplotlib')
    plt = types.ModuleType('matplotlib.pyplot')
    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = plt

    main = importlib.import_module('main')

    calls = []

    class DummyYOLO:
        def __init__(self):
            self.model = object()
            self.callbacks = {}
        def add_callback(self, event, cb):
            pass
        def train(self, *a, **k):
            return {}
        def save(self, path):
            Path(path).write_text('x')
        def to(self, device):
            calls.append(device)

    class DummyPipeline:
        def __init__(self, *a, **k):
            self.model = None
            self.pruning_method = None
            self.metrics_mgr = types.SimpleNamespace(
                to_csv=lambda p: Path(p),
                record_computation=lambda m: None,
            )
        def load_model(self):
            self.model = DummyYOLO()
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
        def save_model(self, path):
            Path(path).write_text('snap')

    class DummyMethod:
        def __init__(self, model=None, workdir=None):
            self.model = model
        def analyze_model(self):
            pass
        def generate_pruning_mask(self, ratio):
            pass
        def apply_pruning(self):
            pass

    monkeypatch.setattr(main, 'PruningPipeline', DummyPipeline)
    monkeypatch.setattr(main, 'YOLO', lambda *a, **k: DummyYOLO())

    cfg = main.TrainConfig(baseline_epochs=0, finetune_epochs=0, batch_size=1, ratios=[0.2], device='cuda:0')
    main.execute_pipeline('m.pt', 'd.yaml', DummyMethod, 0.2, cfg, tmp_path)

    assert calls.count('cuda:0') >= 2

