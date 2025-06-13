import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub heavy dependencies
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch.nn'] = types.ModuleType('torch.nn')

mpl = types.ModuleType('matplotlib')
plt = types.ModuleType('matplotlib.pyplot')
sys.modules['matplotlib'] = mpl
sys.modules['matplotlib.pyplot'] = plt

dummy = types.SimpleNamespace(model=object())
dummy.train = lambda *a, **k: {'mAP': 0.1}

up = types.ModuleType('ultralytics')
up.YOLO = lambda *a, **k: dummy
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda *a, **k: 100
torch_utils.get_num_params = lambda *a, **k: 200
utils.torch_utils = torch_utils
up.utils = utils

sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

import main
import pipeline.pruning_pipeline as pp
pp.YOLO = up.YOLO
pp.get_flops = torch_utils.get_flops
pp.get_num_params = torch_utils.get_num_params


def test_metrics_csv_created(tmp_path):
    cfg = main.TrainConfig(baseline_epochs=1, finetune_epochs=0, batch_size=1, ratios=[0])
    _, csv_path = main.execute_pipeline('m.pt', 'd.yaml', None, 0, cfg, tmp_path)
    assert csv_path.exists()
    header = csv_path.read_text().splitlines()[0]
    assert 'training.mAP' in header
    assert 'pruning.parameters.original' in header
    assert 'pruning.model_size_mb.original' in header
