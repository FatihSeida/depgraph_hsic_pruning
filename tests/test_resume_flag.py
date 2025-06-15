import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_resume_skipped_when_best_only(tmp_path, monkeypatch):
    # Stub heavy dependencies so importing main works
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')

    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_flops = lambda *a, **k: 0
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

    import importlib
    main = importlib.import_module('main')
    class DummyPipeline:
        def __init__(self, *a, **k):
            self.model = types.SimpleNamespace(model=object())
            self.metrics_mgr = types.SimpleNamespace(
                to_csv=lambda p: Path(p),
                record_computation=lambda m: None,
            )
            self.pruning_method = None

        def load_model(self):
            pass

        def calc_initial_stats(self):
            pass

        def pretrain(self, **kw):
            DummyPipeline.kw = kw

        def visualize_results(self):
            pass

        def save_pruning_results(self, path):
            pass

        def save_metrics_csv(self, path):
            return Path(path)

    monkeypatch.setattr(main, "PruningPipeline", DummyPipeline)

    phase_dir = tmp_path / "baseline" / "weights"
    phase_dir.mkdir(parents=True)
    (phase_dir / "best.pt").touch()
    (phase_dir.parent / "results.csv").write_text("epoch\n0\n")

    cfg = main.TrainConfig(baseline_epochs=1, finetune_epochs=0, batch_size=1, ratios=[0])
    main.execute_pipeline("m", "d", None, 0, cfg, tmp_path, resume=True)

    assert DummyPipeline.kw["resume"] is False
