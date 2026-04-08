# tests/dashboard/test_run_manager.py
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.dashboard.run_manager import (
    list_runs, RunInfo, launch_training, get_latest_run_dir,
    kill_training, get_log_lines,
)


@pytest.fixture
def runs_dir(tmp_path):
    for name, steps, best_val in [
        ("run_20260401_100000", [100, 200], 4.3),
        ("run_20260402_120000", [100, 200, 300], 3.9),
    ]:
        d = tmp_path / name
        d.mkdir()
        lines = [
            {"type": "step", "step": s, "train_loss": 4.0, "lr": 3e-4,
             "grad_norm": 0.5, "perplexity": 55.0}
            for s in steps
        ]
        lines.append({"type": "eval", "step": steps[-1], "val_loss": best_val,
                      "val_ppl": 73.0, "eval_train_loss": 3.9})
        lines.append({"type": "best_checkpoint", "step": steps[-1]})
        (d / "metrics.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    return tmp_path


def test_list_runs_returns_all(runs_dir):
    assert len(list_runs(runs_dir)) == 2


def test_list_runs_sorted_newest_first(runs_dir):
    runs = list_runs(runs_dir)
    assert runs[0].name == "run_20260402_120000"
    assert runs[1].name == "run_20260401_100000"


def test_list_runs_run_info_fields(runs_dir):
    newest = list_runs(runs_dir)[0]
    assert isinstance(newest, RunInfo)
    assert newest.last_step == 300
    assert newest.best_val_loss == pytest.approx(3.9)
    assert newest.run_dir == runs_dir / "run_20260402_120000"


def test_list_runs_empty_dir(tmp_path):
    assert list_runs(tmp_path) == []


def test_get_latest_run_dir(runs_dir):
    assert get_latest_run_dir(runs_dir).name == "run_20260402_120000"


def test_get_latest_run_dir_empty(tmp_path):
    assert get_latest_run_dir(tmp_path) is None


@patch("src.dashboard.run_manager.subprocess.Popen")
def test_launch_training_starts_process(mock_popen, tmp_path):
    mock_popen.return_value = MagicMock(pid=1234)
    launch_training(config_path=Path("configs/default.yaml"))
    call_args = mock_popen.call_args[0][0]
    assert "uv" in call_args
    assert "train" in call_args


@patch("src.dashboard.run_manager.subprocess.Popen")
def test_launch_training_with_resume(mock_popen, tmp_path):
    mock_popen.return_value = MagicMock(pid=5678)
    launch_training(
        config_path=Path("configs/default.yaml"),
        resume_run_dir=Path("runs/run_20260402_120000"),
    )
    call_args = mock_popen.call_args[0][0]
    assert "--resume" in call_args


def test_kill_training_terminates():
    proc = MagicMock()
    proc.poll.return_value = None
    kill_training(proc)
    proc.terminate.assert_called_once()


def test_kill_training_already_dead():
    proc = MagicMock()
    proc.poll.return_value = 0
    kill_training(proc)
    proc.terminate.assert_not_called()


def test_get_log_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"line {i}" for i in range(30)))
    lines = get_log_lines(tmp_path, n=5)
    assert lines == ["line 25", "line 26", "line 27", "line 28", "line 29"]


def test_get_log_lines_missing(tmp_path):
    assert get_log_lines(tmp_path) == []
