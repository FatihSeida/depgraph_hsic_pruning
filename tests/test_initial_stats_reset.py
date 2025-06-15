import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# stub heavy deps before importing module
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch.nn'] = types.ModuleType('torch.nn')

up = types.ModuleType('ultralytics')
utils = types.ModuleType('ultralytics.utils')
torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
torch_utils.get_flops = lambda m: 0
torch_utils.get_num_params = lambda m: 0
utils.torch_utils = torch_utils
up.utils = utils
up.YOLO = lambda *a, **k: types.SimpleNamespace(model=object())

sys.modules['ultralytics'] = up
sys.modules['ultralytics.utils'] = utils
sys.modules['ultralytics.utils.torch_utils'] = torch_utils

hsic_mod = types.ModuleType('prune_methods.depgraph_hsic')
class DummyMethod:
    def __init__(self, *a, **k):
        self.reset_called = False
    def reset_records(self):
        self.reset_called = True
hsic_mod.DepgraphHSICMethod = DummyMethod
sys.modules['prune_methods.depgraph_hsic'] = hsic_mod

import pipeline.pruning_pipeline as pp

pp.get_flops = torch_utils.get_flops
pp.get_num_params = torch_utils.get_num_params
pp.count_filters = lambda m: 0
pp.model_size_mb = lambda m: 0


def test_calc_initial_stats_resets_records():
    method = DummyMethod()
    pipeline = pp.PruningPipeline('m', 'd', pruning_method=method)
    pipeline.model = types.SimpleNamespace(model=object())
    pipeline.calc_initial_stats()
    assert method.reset_called
