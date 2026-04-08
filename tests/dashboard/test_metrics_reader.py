# tests/dashboard/test_metrics_reader.py
import json
import time
import pytest
from pathlib import Path
from src.dashboard.metrics_reader import read_metrics, TrainingMetrics, is_metrics_stale


@pytest.fixture
def run_dir(tmp_path):
    lines = [
        {"stage": "train", "type": "model_architecture", "vocab_size": 50257,
         "n_layers": 16, "n_heads": 8, "d_model": 320, "d_ff": 854,
         "total_params": 35763840, "trainable_params": 35763840},
        {"stage": "train", "type": "config", "max_steps": 10000,
         "batch_size": 64, "context_length": 1024, "gradient_accumulation_steps": 4},
        {"stage": "train", "type": "step", "step": 100, "train_loss": 4.5,
         "lr": 3e-4, "perplexity": 90.0, "grad_norm": 0.8, "tokens_per_sec": 12000},
        {"stage": "train", "type": "step", "step": 200, "train_loss": 4.1,
         "lr": 3e-4, "perplexity": 60.0, "grad_norm": 0.7, "tokens_per_sec": 12000},
        {"stage": "train", "type": "eval", "step": 200, "val_loss": 4.3,
         "val_ppl": 73.7, "eval_train_loss": 4.1, "eval_train_ppl": 60.3},
        {"stage": "train", "type": "best_checkpoint", "step": 200},
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    return tmp_path


def test_read_metrics_step_data(run_dir):
    m = read_metrics(run_dir)
    assert m.steps == [100, 200]
    assert m.train_losses == pytest.approx([4.5, 4.1])
    assert m.grad_norms == pytest.approx([0.8, 0.7])
    assert m.lrs == pytest.approx([3e-4, 3e-4])
    assert m.tokens_per_sec == pytest.approx([12000, 12000])


def test_read_metrics_eval_data(run_dir):
    m = read_metrics(run_dir)
    assert m.eval_steps == [200]
    assert m.val_losses == pytest.approx([4.3])
    assert m.val_ppls == pytest.approx([73.7])
    assert m.eval_train_losses == pytest.approx([4.1])


def test_read_metrics_architecture(run_dir):
    m = read_metrics(run_dir)
    assert m.architecture["n_layers"] == 16
    assert m.architecture["total_params"] == 35763840


def test_read_metrics_config(run_dir):
    m = read_metrics(run_dir)
    assert m.config["max_steps"] == 10000
    assert m.config["batch_size"] == 64


def test_read_metrics_best_step(run_dir):
    m = read_metrics(run_dir)
    assert m.best_step == 200


def test_read_metrics_missing_file(tmp_path):
    m = read_metrics(tmp_path)
    assert m.steps == []
    assert m.architecture == {}


def test_read_metrics_partial_fields(tmp_path):
    line = {"stage": "train", "type": "step", "step": 50, "train_loss": 5.0,
            "lr": 1e-4, "perplexity": 148.0, "grad_norm": 1.2}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(line))
    m = read_metrics(tmp_path)
    assert m.steps == [50]
    assert m.tokens_per_sec == []


def test_is_metrics_stale_fresh(run_dir):
    stale, age = is_metrics_stale(run_dir, threshold=60)
    assert stale is False
    assert age < 5


def test_is_metrics_stale_missing(tmp_path):
    stale, age = is_metrics_stale(tmp_path)
    assert stale is False
    assert age == 0
