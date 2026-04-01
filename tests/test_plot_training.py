import sys
from pathlib import Path
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from plot_training import parse_log

SAMPLE_LOG = """
2026-03-20 14:55:19 - INFO - parrotllm.training - Logging initialised -> runs/run_test/train.log
2026-03-20 14:55:29 - INFO - parrotllm.training - step      0 | epoch 0 | loss 10.8774 | lr 2.00e-05 | grad 3.6559
2026-03-20 14:55:29 - DEBUG - parrotllm.training - step      0 | ppl 52964.37 | dt 3.7s
2026-03-20 14:55:49 - INFO - parrotllm.training - step     25 | epoch 0 | loss 9.7866 | lr 5.20e-04 | grad 1.0129
2026-03-20 14:55:49 - DEBUG - parrotllm.training - step     25 | ppl 17793.38 | dt 16.3s
2026-03-20 14:56:37 - INFO - parrotllm.training - step    100 | epoch 0 | loss 7.0557 | lr 1.00e-03 | grad 0.8274
2026-03-20 14:56:37 - DEBUG - parrotllm.training - step    100 | ppl 1159.44 | dt 16.0s
2026-03-20 14:56:37 - INFO - parrotllm.training - Starting evaluation...
2026-03-20 14:56:37 - INFO - parrotllm.training - ------------------------------------------------------------
2026-03-20 14:56:39 - INFO - parrotllm.training -   Train: loss=7.0557, ppl=1159.44
2026-03-20 14:56:39 - INFO - parrotllm.training -   Val:   loss=7.1702, ppl=1300.12
2026-03-20 14:56:39 - INFO - parrotllm.training -   ** New best validation loss! **
2026-03-20 14:56:39 - INFO - parrotllm.training - ------------------------------------------------------------
""".strip()


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "train.log"
    p.write_text(SAMPLE_LOG)
    return p


def test_parse_steps(log_file):
    data = parse_log(log_file)
    assert data["steps"] == [0, 25, 100]


def test_parse_train_loss(log_file):
    data = parse_log(log_file)
    assert data["train_loss"] == pytest.approx([10.8774, 9.7866, 7.0557])


def test_parse_lr(log_file):
    data = parse_log(log_file)
    assert data["lr"] == pytest.approx([2e-05, 5.2e-04, 1e-03])


def test_parse_grad_norm(log_file):
    data = parse_log(log_file)
    assert data["grad_norm"] == pytest.approx([3.6559, 1.0129, 0.8274])


def test_parse_train_ppl(log_file):
    data = parse_log(log_file)
    assert data["train_ppl"] == pytest.approx([52964.37, 17793.38, 1159.44])


def test_parse_eval(log_file):
    data = parse_log(log_file)
    assert data["eval_steps"] == [100]
    assert data["val_loss"] == pytest.approx([7.1702])
    assert data["val_ppl"] == pytest.approx([1300.12])
    assert data["eval_train_loss"] == pytest.approx([7.0557])
    assert data["eval_train_ppl"] == pytest.approx([1159.44])


def test_parse_best_val_step(log_file):
    data = parse_log(log_file)
    assert data["best_val_step"] == 100


def test_parse_label_fallback(log_file):
    data = parse_log(log_file, label="my_run")
    assert data["label"] == "my_run"
