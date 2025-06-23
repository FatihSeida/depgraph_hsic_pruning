import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_calc_stats_records_filters_and_size(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.ModuleType("ultralytics"))
    utils = types.ModuleType("ultralytics.utils")
    torch_utils = types.ModuleType("ultralytics.utils.torch_utils")
    torch_utils.get_flops = lambda *a, **k: 20
    torch_utils.get_num_params = lambda *a, **k: 10
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", torch_utils)

    # stub prune_methods.base to avoid torch dependency
    base_mod = types.ModuleType("prune_methods.base")
    class BasePruningMethod:  # pragma: no cover - placeholder
        pass
    base_mod.BasePruningMethod = BasePruningMethod
    monkeypatch.setitem(sys.modules, "prune_methods.base", base_mod)

    from pipeline.step.calc_stats import CalcStatsStep
    from pipeline.context import PipelineContext

    dummy_model = types.SimpleNamespace(
        model=types.SimpleNamespace(),
        save=lambda p: None,
    )
    ctx = PipelineContext(model_path="m", data="d")
    ctx.model = dummy_model

    monkeypatch.setattr("pipeline.step.calc_stats.get_num_params", lambda m: 10, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.get_flops", lambda m: 20, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters", lambda m: 3, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.file_size_mb", lambda p: 1.5, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_params_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.flops_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_params_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.flops_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_params_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.flops_in_layers", lambda *a, **k: 0, raising=False)

    step = CalcStatsStep("initial")
    step.run(ctx)

    assert ctx.metrics_mgr.pruning["filters"]["original"] == 3
    assert ctx.metrics_mgr.pruning["model_size_mb"]["original"] == 1.5


def test_calc_stats_records_compression_ratio(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.ModuleType("ultralytics"))
    utils = types.ModuleType("ultralytics.utils")
    torch_utils = types.ModuleType("ultralytics.utils.torch_utils")
    torch_utils.get_flops = lambda *a, **k: 20
    torch_utils.get_num_params = lambda *a, **k: 10
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", torch_utils)

    base_mod = types.ModuleType("prune_methods.base")
    class BasePruningMethod:  # pragma: no cover - placeholder
        pass
    base_mod.BasePruningMethod = BasePruningMethod
    monkeypatch.setitem(sys.modules, "prune_methods.base", base_mod)

    from pipeline.step.calc_stats import CalcStatsStep
    from pipeline.context import PipelineContext

    dummy_model = types.SimpleNamespace(
        model=types.SimpleNamespace(),
        save=lambda p: None,
    )
    ctx = PipelineContext(model_path="m", data="d")
    ctx.model = dummy_model

    monkeypatch.setattr("pipeline.step.calc_stats.get_num_params", lambda m: 10, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.get_flops", lambda m: 20, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters", lambda m: 3, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.file_size_mb", lambda p: 1.5, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_params_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.flops_in_layers", lambda *a, **k: 0, raising=False)

    CalcStatsStep("initial").run(ctx)

    monkeypatch.setattr("pipeline.step.calc_stats.get_num_params", lambda m: 5, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.get_flops", lambda m: 10, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters", lambda m: 2, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.file_size_mb", lambda p: 1.0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_params_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.count_filters_in_layers", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr("pipeline.step.calc_stats.flops_in_layers", lambda *a, **k: 0, raising=False)

    CalcStatsStep("pruned").run(ctx)

    assert ctx.metrics_mgr.pruning["compression_ratio"] == 2.0


def test_calc_stats_split_metrics(monkeypatch):
    import torch.nn as nn

    monkeypatch.setitem(sys.modules, "ultralytics", types.ModuleType("ultralytics"))
    utils = types.ModuleType("ultralytics.utils")
    torch_utils = types.ModuleType("ultralytics.utils.torch_utils")
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", torch_utils)

    torch_utils.get_flops = lambda m: sum(1 for mod in m.modules() if isinstance(mod, nn.Conv2d))
    torch_utils.get_num_params = lambda m: sum(p.numel() for p in m.parameters())

    from pipeline.step.calc_stats import CalcStatsStep
    from pipeline.context import PipelineContext

    class DummyYOLO:
        def __init__(self, count):
            self.model = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(count)])

        def save(self, p):
            pass

    ctx = PipelineContext(model_path="m", data="d")
    ctx.model = DummyYOLO(12)

    monkeypatch.setattr("pipeline.step.calc_stats.file_size_mb", lambda p: 0, raising=False)
    step = CalcStatsStep("initial")
    step.run(ctx)

    ctx.model = DummyYOLO(8)
    CalcStatsStep("pruned").run(ctx)

    assert ctx.metrics_mgr.pruning["parameters_backbone"]["original"] == 20
    assert ctx.metrics_mgr.pruning["parameters_head"]["original"] == 4
    assert ctx.metrics_mgr.pruning["parameters_backbone"]["reduction"] == 4
    assert ctx.metrics_mgr.pruning["parameters_head"]["reduction"] == 4
