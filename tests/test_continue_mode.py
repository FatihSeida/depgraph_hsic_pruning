import os
import sys
import types
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

import main


def _prepare_args(tmp_dir, cont):
    return types.SimpleNamespace(
        model="m.pt",
        data="d.yaml",
        runs_dir="experiments",
        methods=["l1"],
        ratios=[0.2],
        baseline_epochs=1,
        finetune_epochs=1,
        batch_size=1,
        device="cuda:0",
        no_baseline=True,
        debug=False,
        cont=cont,
        resume=False,
        compare=True,
    )


def _setup_env(monkeypatch, tmp_path):
    fake_file = tmp_path / "repo" / "main.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(main, "__file__", str(fake_file))
    monkeypatch.setattr(
        main,
        "datetime",
        types.SimpleNamespace(now=lambda: datetime(2023, 1, 1, 0, 0, 0)),
    )
    calls = []

    def fake_exec(*a, **k):
        calls.append((a, k))

    monkeypatch.setattr(main, "execute_pipeline", fake_exec)
    return calls, tmp_path / "experiments" / "pruning_comparison_20230101_000000"


def test_continue_skips_completed_run(monkeypatch, tmp_path):
    calls, base_dir = _setup_env(monkeypatch, tmp_path)
    args = _prepare_args(tmp_path, cont=True)
    monkeypatch.setattr(main, "parse_args", lambda: args)
    run_dir = base_dir / "l1_r0_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".experiment_complete").touch()
    main.main()
    assert calls == []


def test_continue_executes_when_missing(monkeypatch, tmp_path):
    calls, base_dir = _setup_env(monkeypatch, tmp_path)
    args = _prepare_args(tmp_path, cont=True)
    monkeypatch.setattr(main, "parse_args", lambda: args)
    main.main()
    assert len(calls) == 1
    run_dir = base_dir / "l1_r0_2"
    assert (run_dir / ".experiment_complete").exists()
