import os
import sys
import types
import importlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_main_help_shows_options(capsys, monkeypatch):
    # Stub heavy dependencies so importing main works
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')

    up = types.ModuleType('ultralytics_pruning')
    utils = types.ModuleType('ultralytics_pruning.utils')
    torch_utils = types.ModuleType('ultralytics_pruning.utils.torch_utils')
    torch_utils.get_flops = lambda *a, **k: 0
    torch_utils.get_num_params = lambda *a, **k: 0
    utils.torch_utils = torch_utils
    up.utils = utils
    up.YOLO = lambda *a, **k: None
    sys.modules['ultralytics_pruning'] = up
    sys.modules['ultralytics_pruning.utils'] = utils
    sys.modules['ultralytics_pruning.utils.torch_utils'] = torch_utils
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
    ]:
        assert opt in help_text
