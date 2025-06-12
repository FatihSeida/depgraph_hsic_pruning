import os
import sys
import types
import importlib
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_main_help_shows_options(capsys, monkeypatch):
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

    main = importlib.import_module('main')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--help'])
    with pytest.raises(SystemExit):
        main.parse_args()
    help_text = capsys.readouterr().out
    for opt in [
        '--model',
        '--data',
        '--workdir',
        '--resume',
        '--baseline-epochs',
        '--finetune-epochs',
        '--batch-size',
        '--ratios',
        '--device',
    ]:
        assert opt in help_text


def test_device_argument_passed(monkeypatch):
    """--device should be parsed and forwarded to the pipeline."""
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

    main = importlib.import_module('main')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--model', 'm', '--data', 'd', '--device', 'cpu'])
    args = main.parse_args()
    assert args.device == 'cpu'

    calls = {}

    class DummyPipeline:
        def __init__(self, *a, **k):
            self.model = types.SimpleNamespace(model=object())

        def load_model(self):
            pass

        def calc_initial_stats(self):
            pass

        def pretrain(self, **kw):
            calls['pretrain'] = kw

        def visualize_results(self):
            pass

        def save_pruning_results(self, path):
            pass

    monkeypatch.setattr(main, 'PruningPipeline', DummyPipeline)
    main.execute_pipeline('m', 'd', None, 0.2, main.TrainConfig(device=args.device), Path('w'))
    assert calls['pretrain']['device'] == 'cpu'

