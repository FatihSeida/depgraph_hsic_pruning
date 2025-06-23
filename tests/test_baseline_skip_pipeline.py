import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_pipeline_runs_without_baseline(monkeypatch, tmp_path):
    # stub torch
    class DummyTensor:
        def unsqueeze(self, *a, **k):
            return self
        def to(self, device):
            return self
        def numel(self):
            return 1

    torch = types.ModuleType('torch')
    torch.tensor = lambda *a, **k: DummyTensor()
    torch.randn = lambda *a, **k: DummyTensor()
    class _NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
    torch.no_grad = lambda: _NoGrad()
    torch.nn = types.ModuleType('torch.nn')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.nn', torch.nn)

    # stub ultralytics
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = torch_utils

    class DummyModel:
        def __call__(self, x):
            pass
        def parameters(self):
            return iter([types.SimpleNamespace(device='cpu')])

    class DummyYOLO:
        def __init__(self, *a, **k):
            self.model = DummyModel()
            self.callbacks = {}
        def add_callback(self, event, cb):
            self.callbacks.setdefault(event, []).append(cb)
        def train(self, *a, **k):
            return {}

    up.utils = utils
    up.YOLO = lambda *a, **k: DummyYOLO()
    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    # stub PIL, numpy and yaml
    pil = types.ModuleType('PIL')
    class DummyImage:
        size = (640, 640)
        def convert(self, mode):
            return self
        def resize(self, size):
            self.size = size
            return self
    class ImgMod:
        @staticmethod
        def open(path):
            return DummyImage()
    pil.Image = ImgMod
    monkeypatch.setitem(sys.modules, 'PIL', pil)

    np = types.ModuleType('numpy')
    np.array = lambda *a, **k: [[0]]
    np.transpose = lambda arr, axes: arr
    np.float32 = float
    monkeypatch.setitem(sys.modules, 'numpy', np)

    # stub pandas to avoid heavy import
    pd = types.ModuleType('pandas')
    monkeypatch.setitem(sys.modules, 'pandas', pd)

    yaml_mod = types.ModuleType('yaml')
    yaml_mod.safe_load = lambda f: {"path": str(Path(f.name).parent), "val": "images"}
    monkeypatch.setitem(sys.modules, 'yaml', yaml_mod)

    # stub pruning method
    hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')
    class DummyMethod:
        def __init__(self, model=None, workdir=None, **kw):
            self.model = model
            DummyMethod.calls = []
        def analyze_model(self):
            DummyMethod.calls.append('analyze')
        def generate_pruning_mask(
            self, ratio, dataloader=None
        ):
            DummyMethod.calls.append('mask')
        def apply_pruning(self):
            DummyMethod.calls.append('apply')
        def add_labels(self, labels):
            DummyMethod.calls.append('labels')
        example_inputs = DummyTensor()
    hsic_mod.DepgraphHSICMethod = DummyMethod
    monkeypatch.setitem(sys.modules, 'prune_methods.depgraph_hsic', hsic_mod)

    import importlib
    main = importlib.import_module('main')

    class DummyPipeline:
        def __init__(self, *a, **k):
            self.model = DummyYOLO()
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
            DummyMethod.calls.append('pipeline_analyze')
        def set_pruning_method(self, method):
            self.pruning_method = method
        def generate_pruning_mask(self, ratio, dataloader=None):
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
            Path(path).write_text('x')

    monkeypatch.setattr(main, 'PruningPipeline', DummyPipeline)

    (tmp_path / 'images').mkdir()
    (tmp_path / 'images' / 'img.jpg').write_text('x')
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'labels' / 'img.txt').write_text('0')
    data_file = tmp_path / 'd.yaml'
    data_file.write_text('path: .\nval: images')

    cfg = main.TrainConfig(baseline_epochs=0, finetune_epochs=0, batch_size=1, ratios=[0.2])
    main.execute_pipeline('m.pt', str(data_file), DummyMethod, 0.2, cfg, tmp_path)

    assert DummyMethod.calls.count('pipeline_analyze') == 1
