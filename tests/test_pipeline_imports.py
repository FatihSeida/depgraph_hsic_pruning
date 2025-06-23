import os
import sys
import types
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_step_imports_via_pipeline():
    # Stub heavy dependencies
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')
    up = types.ModuleType('ultralytics')
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 0
    from helper import flops_utils as fu
    monkeypatch.setattr(fu, "get_flops_reliable", lambda *a, **k: 0, raising=False)
    utils.torch_utils = torch_utils
    up.utils = utils
    up.YOLO = lambda *a, **k: None
    sys.modules['ultralytics'] = up
    sys.modules['ultralytics.utils'] = utils
    sys.modules['ultralytics.utils.torch_utils'] = torch_utils
    base = types.ModuleType('prune_methods.base')
    class BasePruningMethod:  # pragma: no cover - placeholder
        pass
    base.BasePruningMethod = BasePruningMethod
    sys.modules['prune_methods.base'] = base

    pipeline = importlib.import_module('pipeline')

    mapping = {
        'LoadModelStep': 'load_model',
        'TrainStep': 'train',
        'AnalyzeModelStep': 'analyze',
        'AnalyzeAfterTrainingStep': 'analyze',
        'GenerateMasksStep': 'generate_masks',
        'ApplyPruningStep': 'apply_pruning',
        'ReconfigureModelStep': 'reconfigure',
        'CalcStatsStep': 'calc_stats',
        'CompareModelsStep': 'compare',
    }

    for cls_name, mod_name in mapping.items():
        cls = getattr(pipeline, cls_name)
        mod = importlib.import_module(f'pipeline.step.{mod_name}')
        assert getattr(mod, cls_name) is cls
