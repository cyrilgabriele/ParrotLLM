"""Tests for the logging utilities."""

import json
import logging
import os
import tempfile

from src.logging_utils import JSONLLogger, make_run_dir, setup_logger


def test_init_logging_console_only():
    """init_logging sets up console handler on root parrotllm logger."""
    from src.logging_utils import init_logging

    log = init_logging(console_level="WARNING")
    assert log.name == "parrotllm"
    console_handlers = [h for h in log.handlers if isinstance(h, logging.StreamHandler)
                        and not isinstance(h, logging.FileHandler)]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.WARNING
    log.handlers.clear()


def test_setup_logger_creates_log_file():
    with tempfile.TemporaryDirectory() as tmp:
        log = setup_logger(tmp, console_level="WARNING", file_level="DEBUG")
        log.info("hello from test")

        log_path = os.path.join(tmp, "train.log")
        assert os.path.exists(log_path)

        content = open(log_path).read()
        assert "hello from test" in content
        assert "- INFO -" in content

        # cleanup handlers to avoid interference between tests
        log.handlers.clear()


def test_logger_component_levels():
    with tempfile.TemporaryDirectory() as tmp:
        log = setup_logger(
            tmp,
            console_level="WARNING",
            file_level="DEBUG",
            component_levels={"model_initialization": "WARNING"},
        )
        model_log = logging.getLogger("parrotllm.model_initialization")

        model_log.info("should be suppressed")
        model_log.warning("should appear")

        log_path = os.path.join(tmp, "train.log")
        content = open(log_path).read()
        assert "should be suppressed" not in content
        assert "should appear" in content

        log.handlers.clear()


def test_jsonl_logger_writes_valid_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        jlog = JSONLLogger(tmp)
        jlog.log("pretraining", "step", epoch=0, step=1, train_loss=10.5, lr=6e-4)
        jlog.log("pretraining", "step", epoch=0, step=2, train_loss=10.3, lr=6e-4)
        jlog.log("pretraining", "eval", step=2, val_loss=10.4, val_ppl=33000.0)
        jlog.close()

        path = os.path.join(tmp, "metrics.jsonl")
        assert os.path.exists(path)

        lines = open(path).readlines()
        assert len(lines) == 3

        for line in lines:
            record = json.loads(line)
            assert "stage" in record
            assert "type" in record
            assert "timestamp" in record

        step_record = json.loads(lines[0])
        assert step_record["stage"] == "pretraining"
        assert step_record["type"] == "step"
        assert step_record["train_loss"] == 10.5

        eval_record = json.loads(lines[2])
        assert eval_record["type"] == "eval"
        assert eval_record["val_loss"] == 10.4


def test_make_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = make_run_dir(tmp, tag="test")
        assert os.path.isdir(run_dir)
        assert "run_" in os.path.basename(run_dir)
        assert "_test" in os.path.basename(run_dir)


def test_make_run_dir_without_tag():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = make_run_dir(tmp)
        assert os.path.isdir(run_dir)
        assert "_test" not in os.path.basename(run_dir)
