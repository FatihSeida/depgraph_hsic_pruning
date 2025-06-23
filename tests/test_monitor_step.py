import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.step.monitor_computation as mc


class DummyGPU:
    def __init__(self):
        self.calls = 0
    def reset(self):
        pass
    def collect(self, **kw):
        self.calls += 1
        return {
            'gpu_utilization': 42,
            'gpu_memory_used_mb': 1,
            'gpu_memory_total_mb': 2,
            'gpu_memory_percent': 50,
        }
    def get_summary(self):
        return {'avg_utilization': 42, 'avg_memory_percent': 50, 'count': self.calls}


class DummyMemory:
    def __init__(self):
        self.calls = 0
    def reset(self):
        pass
    def collect(self, **kw):
        self.calls += 1
        return {'ram_used_mb': 3, 'ram_total_mb': 4, 'ram_percent': 30}
    def get_summary(self):
        return {'avg_ram_percent': 30, 'peak_ram_percent': 30, 'count': self.calls}


class DummyPower:
    def __init__(self):
        self.calls = 0
    def reset(self):
        pass
    def collect(self, interval=None, **kw):
        self.calls += 1
        return {
            'power_usage_watts': 5,
            'energy_joules': 0,
            'total_energy_joules': 0,
            'total_energy_wh': 0,
        }
    def get_summary(self):
        return {
            'avg_power_watts': 5,
            'peak_power_watts': 5,
            'total_energy_wh': 0,
            'collection_duration_s': self.calls,
        }


def test_monitor_metrics_recorded(monkeypatch, tmp_path):
    # stub heavy deps
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setitem(sys.modules, 'torch.nn', types.ModuleType('torch.nn'))
    mpl = types.ModuleType('matplotlib')
    plt = types.ModuleType('matplotlib.pyplot')
    monkeypatch.setitem(sys.modules, 'matplotlib', mpl)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', plt)

    dummy = types.SimpleNamespace(model=object())
    dummy.train = lambda *a, **k: {}

    up = types.ModuleType('ultralytics')
    up.YOLO = lambda *a, **k: dummy
    utils = types.ModuleType('ultralytics.utils')
    torch_utils = types.ModuleType('ultralytics.utils.torch_utils')
    torch_utils.get_num_params = lambda *a, **k: 20
    utils.torch_utils = torch_utils
    from helper import flops_utils as fu
    fu.get_flops_reliable = lambda *a, **k: 10
    up.utils = utils

    monkeypatch.setitem(sys.modules, 'ultralytics', up)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils', utils)
    monkeypatch.setitem(sys.modules, 'ultralytics.utils.torch_utils', torch_utils)

    import main
    import pipeline.pruning_pipeline as pp
    monkeypatch.setattr(pp, 'YOLO', up.YOLO, raising=False)
    monkeypatch.setattr(pp, 'get_flops_reliable', fu.get_flops_reliable, raising=False)
    monkeypatch.setattr(pp, 'get_num_params_reliable', torch_utils.get_num_params, raising=False)

    monkeypatch.setattr(mc, 'GPUMetric', DummyGPU)
    monkeypatch.setattr(mc, 'MemoryMetric', DummyMemory)
    monkeypatch.setattr(mc, 'PowerMetric', DummyPower)

    cfg = main.TrainConfig(baseline_epochs=1, finetune_epochs=0, batch_size=1, ratios=[0])
    pipeline, _ = main.execute_pipeline('m.pt', 'd.yaml', None, 0, cfg, tmp_path)
    comp = pipeline.metrics_mgr.computation
    assert comp['gpu_utilization'] == 42
    assert comp['gpu_memory_percent'] == 50
    assert comp['ram_percent'] == 30
    assert comp['avg_ram_used_mb'] == 3
    assert comp['power_usage_watts'] == 5
    metrics = pipeline.record_metrics()
    assert 'computation' in metrics
    assert metrics['computation']['gpu_utilization'] == 42
    assert metrics['computation']['gpu_memory_percent'] == 50
    assert metrics['computation']['ram_percent'] == 30
    assert metrics['computation']['avg_ram_used_mb'] == 3
    assert metrics['computation']['power_usage_watts'] == 5
