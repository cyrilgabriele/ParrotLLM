"""Tests for the PyTorch-based training LR scheduler."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from configs import ProjectConfig, TrainingConfig
from src.training.trainer import (
    ParrotLRScheduler,
    _apply_optimizer_step,
    load_checkpoint,
    run_train,
    save_checkpoint,
)


def _build_scheduler(*, schedule: str) -> tuple[torch.nn.Linear, torch.optim.AdamW, ParrotLRScheduler]:
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = ParrotLRScheduler(
        optimizer,
        warmup_steps=5,
        max_steps=20,
        min_lr=0.1,
        schedule=schedule,
        decay_ratio=0.1,
    )
    return model, optimizer, scheduler


def _collect_step_lrs(
    optimizer: torch.optim.Optimizer,
    scheduler: ParrotLRScheduler,
    *,
    max_steps: int,
) -> list[float]:
    lrs = []
    for _ in range(max_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_wsd_scheduler_matches_expected_warmup_and_reaches_min_lr():
    _, optimizer, scheduler = _build_scheduler(schedule="wsd")

    lrs = _collect_step_lrs(optimizer, scheduler, max_steps=20)

    assert lrs[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert lrs[17] == pytest.approx(1.0)
    assert lrs[18] == pytest.approx(0.55)
    assert lrs[19] == pytest.approx(0.1)


def test_cosine_scheduler_matches_expected_warmup_and_reaches_min_lr():
    _, optimizer, scheduler = _build_scheduler(schedule="cosine")

    lrs = _collect_step_lrs(optimizer, scheduler, max_steps=20)

    assert lrs[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert lrs[5] < lrs[4]
    assert lrs[-1] == pytest.approx(0.1)


def test_checkpoint_round_trip_restores_scheduler_state(tmp_path: Path):
    model, optimizer, scheduler = _build_scheduler(schedule="wsd")
    for _ in range(7):
        optimizer.step()
        scheduler.step()

    expected_step = 7
    expected_lr = optimizer.param_groups[0]["lr"]
    path = save_checkpoint(
        model,
        optimizer,
        {"training": {"learning_rate": 1.0}},
        step=expected_step,
        epoch=0,
        scaler=None,
        checkpoint_dir=str(tmp_path),
        filename="scheduler.pt",
        scheduler=scheduler,
    )

    restored_model, restored_optimizer, restored_scheduler = _build_scheduler(schedule="wsd")
    loaded_step, _ = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        scaler=None,
        device=torch.device("cpu"),
        scheduler=restored_scheduler,
    )

    assert loaded_step == expected_step
    assert restored_scheduler.last_epoch == expected_step
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)

    optimizer.step()
    scheduler.step()
    restored_optimizer.step()
    restored_scheduler.step()

    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_checkpoint_round_trip_restores_trainer_state(tmp_path: Path):
    model, optimizer, scheduler = _build_scheduler(schedule="wsd")
    trainer_state = {"next_epoch": 2, "next_micro_batch": 1, "data_seed": 42}
    path = save_checkpoint(
        model,
        optimizer,
        {"training": {"learning_rate": 1.0}},
        step=3,
        epoch=1,
        scaler=None,
        checkpoint_dir=str(tmp_path),
        filename="trainer_state.pt",
        scheduler=scheduler,
        trainer_state=trainer_state,
    )

    restored_model, restored_optimizer, restored_scheduler = _build_scheduler(schedule="wsd")
    loaded_step, _, loaded_trainer_state = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        scaler=None,
        device=torch.device("cpu"),
        scheduler=restored_scheduler,
        return_trainer_state=True,
    )

    assert loaded_step == 3
    assert loaded_trainer_state == trainer_state


def test_loading_legacy_checkpoint_reconstructs_scheduler_position(tmp_path: Path):
    model, optimizer, scheduler = _build_scheduler(schedule="cosine")
    for _ in range(4):
        optimizer.step()
        scheduler.step()

    expected_step = 4
    expected_lr = optimizer.param_groups[0]["lr"]
    path = save_checkpoint(
        model,
        optimizer,
        {"training": {"learning_rate": 1.0}},
        step=expected_step,
        epoch=0,
        scaler=None,
        checkpoint_dir=str(tmp_path),
        filename="legacy.pt",
    )

    restored_model, restored_optimizer, restored_scheduler = _build_scheduler(schedule="cosine")
    loaded_step, _ = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        scaler=None,
        device=torch.device("cpu"),
        scheduler=restored_scheduler,
    )

    assert loaded_step == expected_step
    assert restored_scheduler.last_epoch == expected_step
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_apply_optimizer_step_reports_when_grad_scaler_skips():
    class CountingOptimizer:
        def __init__(self):
            self.step_calls = 0

        def step(self):
            self.step_calls += 1

    class FakeScaler:
        def __init__(self, *, skip_step: bool):
            self._skip_step = skip_step
            self._scale = 1024.0

        def get_scale(self):
            return self._scale

        def step(self, optimizer):
            if not self._skip_step:
                optimizer.step()

        def update(self):
            if self._skip_step:
                self._scale /= 2.0

    skipped_optimizer = CountingOptimizer()
    assert _apply_optimizer_step(skipped_optimizer, FakeScaler(skip_step=True)) is False
    assert skipped_optimizer.step_calls == 0

    updated_optimizer = CountingOptimizer()
    assert _apply_optimizer_step(updated_optimizer, FakeScaler(skip_step=False)) is True
    assert updated_optimizer.step_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr_schedule", "linear"),
        ("lr_decay_ratio", 1.5),
    ],
)
def test_training_config_rejects_invalid_scheduler_fields(field: str, value):
    payload = {
        "device": "cpu",
        "train_bin": "train.bin",
        "val_bin": "val.bin",
        "num_workers": 0,
        "pin_memory": False,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "warmup_steps": 0,
        "max_steps": 10,
        "lr_schedule": "wsd",
        "lr_decay_ratio": 0.1,
        "z_loss_coeff": 0.0,
        "save_every": 10,
        "eval_every": 10,
        "checkpoint_dir": "checkpoints",
        "keep_last_checkpoints": 1,
        "keep_best_checkpoints": 1,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "runs_dir": "runs",
        "log_every": 1,
        "compile": False,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


def test_training_config_rejects_min_lr_above_learning_rate():
    payload = {
        "device": "cpu",
        "train_bin": "train.bin",
        "val_bin": "val.bin",
        "num_workers": 0,
        "pin_memory": False,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "min_lr": 2e-3,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "warmup_steps": 0,
        "max_steps": 10,
        "lr_schedule": "wsd",
        "lr_decay_ratio": 0.1,
        "z_loss_coeff": 0.0,
        "save_every": 10,
        "eval_every": 10,
        "checkpoint_dir": "checkpoints",
        "keep_last_checkpoints": 1,
        "keep_best_checkpoints": 1,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "runs_dir": "runs",
        "log_every": 1,
        "compile": False,
    }

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


def _write_token_file(path: Path, *, vocab_size: int, n_chunks: int, context_length: int) -> None:
    token_count = n_chunks * (context_length + 1)
    tokens = (np.arange(token_count, dtype=np.uint16) % vocab_size).astype(np.uint16)
    tokens.tofile(path)


def _build_tiny_project_config(tmp_path: Path, *, runs_dir: Path) -> ProjectConfig:
    train_bin = tmp_path / "train.bin"
    _write_token_file(train_bin, vocab_size=32, n_chunks=7, context_length=4)

    payload = {
        "model": {
            "vocab_size": 32,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 16,
            "context_length": 4,
            "bias": False,
            "dropout": 0.0,
            "rope_theta": 10000.0,
            "gradient_checkpointing": False,
        },
        "training": {
            "device": "cpu",
            "train_bin": str(train_bin),
            "val_bin": str(tmp_path / "missing-val.bin"),
            "num_workers": 0,
            "pin_memory": False,
            "batch_size": 2,
            "gradient_accumulation_steps": 3,
            "learning_rate": 1e-2,
            "min_lr": 1e-3,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "warmup_steps": 1,
            "max_steps": 3,
            "lr_schedule": "wsd",
            "lr_decay_ratio": 0.1,
            "z_loss_coeff": 0.0,
            "save_every": 1,
            "eval_every": 100,
            "checkpoint_dir": "checkpoints",
            "keep_last_checkpoints": 5,
            "keep_best_checkpoints": 0,
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.0,
            "runs_dir": str(runs_dir),
            "log_every": 1,
            "compile": False,
        },
        "logging": {
            "console_level": "WARNING",
            "file_level": "WARNING",
            "components": {"training": "WARNING"},
        },
    }
    return ProjectConfig.model_validate(payload)


def _single_checkpoint(run_root: Path, pattern: str) -> Path:
    matches = list(run_root.glob(f"run_*/checkpoints/{pattern}"))
    assert len(matches) == 1
    return matches[0]


def test_mid_epoch_resume_restores_exact_data_position(tmp_path: Path):
    full_config = _build_tiny_project_config(tmp_path, runs_dir=tmp_path / "runs-full")
    model_config = full_config.model_dump(mode="python")
    run_train(full_config, model_config, device=torch.device("cpu"))

    full_step1_ckpt = _single_checkpoint(Path(full_config.training.runs_dir), "*step_0000001.pt")
    full_final_ckpt = _single_checkpoint(Path(full_config.training.runs_dir), "*step_0000003.pt")

    resume_config = _build_tiny_project_config(tmp_path, runs_dir=tmp_path / "runs-resume")
    run_train(
        resume_config,
        resume_config.model_dump(mode="python"),
        device=torch.device("cpu"),
        checkpoint=str(full_step1_ckpt),
    )
    resumed_final_ckpt = _single_checkpoint(
        Path(resume_config.training.runs_dir),
        "*step_0000003.pt",
    )

    full_state = torch.load(full_final_ckpt, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed_final_ckpt, map_location="cpu", weights_only=False)

    assert full_state["step"] == resumed_state["step"] == 3
    assert full_state["trainer_state"] == resumed_state["trainer_state"]
    assert full_state["scheduler"]["last_epoch"] == resumed_state["scheduler"]["last_epoch"]
    for key, tensor in full_state["model"].items():
        torch.testing.assert_close(tensor, resumed_state["model"][key])
