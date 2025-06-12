import os
import sys
import types
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pruning_comparison


def _prepare_args(tmp_dir, cont):
    return types.SimpleNamespace(
        model="m.pt",
        data="d.yaml",
        device="cpu",
        runs_dir="experiments",
        methods=["l1"],
        ratios=[0.2],
        baseline_epochs=1,
        no_baseline=True,
        debug=False,
        cont=cont,
    )


def _setup_env(monkeypatch, tmp_path):
    fake_file = tmp_path / "repo" / "pruning_comparison.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(pruning_comparison, "__file__", str(fake_file))
    monkeypatch.setattr(
        pruning_comparison,
        "datetime",
        types.SimpleNamespace(now=lambda: datetime(2023, 1, 1, 0, 0, 0)),
    )
    calls = []

    def fake_run(cmd, debug=False):
        calls.append(cmd)

    monkeypatch.setattr(pruning_comparison, "run_command", fake_run)
    return calls, tmp_path / "experiments" / "pruning_comparison_20230101_000000"


def test_continue_skips_completed_run(monkeypatch, tmp_path):
    calls, base_dir = _setup_env(monkeypatch, tmp_path)
    args = _prepare_args(tmp_path, cont=True)
    monkeypatch.setattr(pruning_comparison, "parse_args", lambda: args)
    run_dir = base_dir / "l1_r0_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".experiment_complete").touch()
    pruning_comparison.main()
    assert calls == []


def test_continue_executes_when_missing(monkeypatch, tmp_path):
    calls, base_dir = _setup_env(monkeypatch, tmp_path)
    args = _prepare_args(tmp_path, cont=True)
    monkeypatch.setattr(pruning_comparison, "parse_args", lambda: args)
    pruning_comparison.main()
    assert len(calls) == 1
    run_dir = base_dir / "l1_r0_2"
    assert (run_dir / ".experiment_complete").exists()
