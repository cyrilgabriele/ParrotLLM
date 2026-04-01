import json
import sys
from pathlib import Path
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scripts"))
from plot_training import parse_log, _resolve_run

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


def test_parse_label_directory_fallback(log_file):
    data = parse_log(log_file)  # no config.json, no explicit label
    assert data["label"] == log_file.parent.name


def test_parse_label_from_config(log_file):
    config = {
        "training": {"learning_rate": 0.001},
        "model": {"n_layers": 4, "d_model": 128}
    }
    config_path = log_file.parent / "config.json"
    config_path.write_text(json.dumps(config))
    data = parse_log(log_file)
    assert data["label"] == "lr=0.001, layers=4, d=128"


def test_build_figure_single_run(log_file):
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for tests
    from plot_training import build_figure
    data = parse_log(log_file)
    fig = build_figure([data])
    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) == 5  # 4 subplots + 1 twinx for LR/grad subplot


def test_build_figure_comparison(log_file):
    import matplotlib
    matplotlib.use("Agg")
    from plot_training import build_figure
    data1 = parse_log(log_file, label="run_A")
    data2 = parse_log(log_file, label="run_B")
    fig = build_figure([data1, data2])
    assert fig is not None
    assert len(fig.get_axes()) == 5  # 4 subplots + 1 twinx


def test_resolve_run_missing_log(tmp_path):
    with pytest.raises(FileNotFoundError, match="No train.log found"):
        _resolve_run(tmp_path)
