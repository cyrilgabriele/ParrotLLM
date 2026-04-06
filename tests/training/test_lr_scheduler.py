"""Tests for LR scheduling and scheduler-aware trainer resume semantics."""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from configs import ProjectConfig, TrainingConfig
from src.training.trainer import (
    ParrotLRScheduler,
    _apply_optimizer_step,
    compute_lr,
    load_checkpoint,
    run_train,
    save_checkpoint,
)


def _build_scheduler(
    *,
    schedule: str,
    base_lrs: tuple[float, ...] = (1.0,),
    warmup_steps: int = 5,
    max_steps: int = 20,
    min_lr: float = 0.1,
    decay_ratio: float = 0.1,
) -> tuple[torch.nn.ModuleList, torch.optim.AdamW, ParrotLRScheduler]:
    model = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in base_lrs])
    optimizer = torch.optim.AdamW(
        [
            {"params": layer.parameters(), "lr": lr}
            for layer, lr in zip(model, base_lrs)
        ]
    )
    scheduler = ParrotLRScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
        schedule=schedule,
        decay_ratio=decay_ratio,
    )
    return model, optimizer, scheduler


def _current_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def _set_deterministic_grads(model: torch.nn.Module) -> None:
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, fill_value=0.01 * index)


def _step_optimizer_and_scheduler(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ParrotLRScheduler,
    *,
    steps: int,
) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _set_deterministic_grads(model)
        optimizer.step()
        scheduler.step()


def _assert_optimizer_states_equal(
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for param_id in left["state"]:
        left_state = left["state"][param_id]
        right_state = right["state"][param_id]
        assert left_state.keys() == right_state.keys()
        for key, left_value in left_state.items():
            right_value = right_state[key]
            if torch.is_tensor(left_value):
                torch.testing.assert_close(left_value, right_value)
            else:
                assert left_value == right_value


@pytest.mark.parametrize("schedule", ["wsd", "cosine"])
def test_scheduler_tracks_closed_form_lr_for_each_step_and_param_group(schedule: str):
    _, optimizer, scheduler = _build_scheduler(
        schedule=schedule,
        base_lrs=(1.0, 0.4),
        warmup_steps=3,
        max_steps=9,
        min_lr=0.1,
        decay_ratio=1 / 3,
    )

    for step in range(12):
        expected = [
            compute_lr(
                step,
                warmup_steps=3,
                max_steps=9,
                max_lr=base_lr,
                min_lr=0.1,
                schedule=schedule,
                decay_ratio=1 / 3,
            )
            for base_lr in (1.0, 0.4)
        ]
        assert _current_lrs(optimizer) == pytest.approx(expected)
        optimizer.step()
        scheduler.step()


def test_wsd_schedule_has_expected_warmup_plateau_decay_and_floor():
    _, optimizer, scheduler = _build_scheduler(schedule="wsd")

    lrs = []
    for _ in range(23):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert lrs[5:18] == pytest.approx([1.0] * 13)
    assert lrs[18] == pytest.approx(0.55)
    assert lrs[19:] == pytest.approx([0.1] * 4)


def test_cosine_schedule_decays_after_warmup_and_stays_at_floor():
    _, optimizer, scheduler = _build_scheduler(schedule="cosine")

    lrs = []
    for _ in range(23):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert all(previous > current for previous, current in zip(lrs[4:19], lrs[5:20]))
    assert lrs[19] == pytest.approx(0.1)
    assert lrs[20:] == pytest.approx([0.1] * 3)


def test_wsd_decay_is_clamped_to_the_post_warmup_window():
    values = [
        compute_lr(
            step,
            warmup_steps=8,
            max_steps=10,
            max_lr=1.0,
            min_lr=0.2,
            schedule="wsd",
            decay_ratio=1.0,
        )
        for step in range(10)
    ]

    assert values[:8] == pytest.approx([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    assert values[8] == pytest.approx(0.6)
    assert values[9] == pytest.approx(0.2)


@pytest.mark.parametrize("schedule", ["wsd", "cosine"])
def test_resume_to_step_restores_current_lr_and_future_progression(schedule: str):
    model, optimizer, scheduler = _build_scheduler(
        schedule=schedule,
        warmup_steps=3,
        max_steps=10,
        min_lr=0.1,
        decay_ratio=0.3,
    )
    _step_optimizer_and_scheduler(model, optimizer, scheduler, steps=4)
    expected_current_lrs = _current_lrs(optimizer)
    expected_future_lrs = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        expected_future_lrs.append(_current_lrs(optimizer))

    _, restored_optimizer, restored_scheduler = _build_scheduler(
        schedule=schedule,
        warmup_steps=3,
        max_steps=10,
        min_lr=0.1,
        decay_ratio=0.3,
    )
    restored_scheduler.resume_to_step(4)

    assert restored_scheduler.last_epoch == 4
    assert restored_scheduler._step_count == 5
    assert _current_lrs(restored_optimizer) == pytest.approx(expected_current_lrs)

    for expected_lrs in expected_future_lrs:
        restored_optimizer.step()
        restored_scheduler.step()
        assert _current_lrs(restored_optimizer) == pytest.approx(expected_lrs)


@pytest.mark.parametrize("schedule", ["wsd", "cosine"])
def test_checkpoint_round_trip_restores_scheduler_state_and_future_progression(
    tmp_path: Path,
    schedule: str,
):
    model, optimizer, scheduler = _build_scheduler(
        schedule=schedule,
        base_lrs=(1.0, 0.4),
        warmup_steps=3,
        max_steps=10,
        min_lr=0.1,
        decay_ratio=0.3,
    )
    _step_optimizer_and_scheduler(model, optimizer, scheduler, steps=6)

    expected_scheduler_state = copy.deepcopy(scheduler.state_dict())
    expected_optimizer_state = copy.deepcopy(optimizer.state_dict())
    expected_current_lrs = _current_lrs(optimizer)

    path = save_checkpoint(
        model,
        optimizer,
        {"training": {"learning_rate": 1.0}},
        step=6,
        epoch=0,
        scaler=None,
        checkpoint_dir=str(tmp_path),
        filename=f"{schedule}.pt",
        scheduler=scheduler,
    )

    expected_future_lrs = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        expected_future_lrs.append(_current_lrs(optimizer))

    restored_model, restored_optimizer, restored_scheduler = _build_scheduler(
        schedule=schedule,
        base_lrs=(1.0, 0.4),
        warmup_steps=3,
        max_steps=10,
        min_lr=0.1,
        decay_ratio=0.3,
    )
    loaded_step, _ = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        scaler=None,
        device=torch.device("cpu"),
        scheduler=restored_scheduler,
    )

    assert loaded_step == 6
    assert restored_scheduler.state_dict() == expected_scheduler_state
    _assert_optimizer_states_equal(expected_optimizer_state, restored_optimizer.state_dict())
    assert _current_lrs(restored_optimizer) == pytest.approx(expected_current_lrs)

    for expected_lrs in expected_future_lrs:
        restored_optimizer.step()
        restored_scheduler.step()
        assert _current_lrs(restored_optimizer) == pytest.approx(expected_lrs)


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


def test_checkpoint_round_trip_restores_rng_state(tmp_path: Path):
    model, optimizer, scheduler = _build_scheduler(schedule="wsd")

    random.seed(1234)
    np.random.seed(5678)
    torch.manual_seed(91011)

    _ = random.random()
    _ = np.random.random()
    _ = torch.rand(1)

    path = save_checkpoint(
        model,
        optimizer,
        {"training": {"learning_rate": 1.0}},
        step=1,
        epoch=0,
        scaler=None,
        checkpoint_dir=str(tmp_path),
        filename="rng.pt",
        scheduler=scheduler,
    )

    expected_python = random.random()
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    _ = random.random()
    _ = np.random.random()
    _ = torch.rand(1)

    restored_model, restored_optimizer, restored_scheduler = _build_scheduler(schedule="wsd")
    load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        scaler=None,
        device=torch.device("cpu"),
        scheduler=restored_scheduler,
    )

    assert random.random() == pytest.approx(expected_python)
    assert np.random.random(3) == pytest.approx(expected_numpy)
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_loading_legacy_checkpoint_fast_forwards_scheduler_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    model, optimizer, scheduler = _build_scheduler(
        schedule="cosine",
        warmup_steps=3,
        max_steps=10,
        min_lr=0.1,
        decay_ratio=0.3,
    )
    _step_optimizer_and_scheduler(model, optimizer, scheduler, steps=4)
    expected_current_lrs = _current_lrs(optimizer)

    path = save_checkpoint(
        model,
        optimizer,
        {"training": {"learning_rate": 1.0}},
        step=4,
        epoch=0,
        scaler=None,
        checkpoint_dir=str(tmp_path),
        filename="legacy.pt",
    )

    expected_future_lrs = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        expected_future_lrs.append(_current_lrs(optimizer))

    restored_model, restored_optimizer, restored_scheduler = _build_scheduler(
        schedule="cosine",
        warmup_steps=3,
        max_steps=10,
        min_lr=0.1,
        decay_ratio=0.3,
    )
    with caplog.at_level(logging.WARNING, logger="parrotllm.training"):
        loaded_step, _ = load_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            scaler=None,
            device=torch.device("cpu"),
            scheduler=restored_scheduler,
        )

    assert loaded_step == 4
    assert restored_scheduler.last_epoch == 4
    assert "has no scheduler state; reconstructing the LR position" in caplog.text
    assert _current_lrs(restored_optimizer) == pytest.approx(expected_current_lrs)

    for expected_lrs in expected_future_lrs:
        restored_optimizer.step()
        restored_scheduler.step()
        assert _current_lrs(restored_optimizer) == pytest.approx(expected_lrs)


def test_apply_optimizer_step_only_advances_scheduler_on_real_update():
    _, optimizer, scheduler = _build_scheduler(
        schedule="wsd",
        warmup_steps=5,
        max_steps=20,
        min_lr=0.1,
    )
    initial_lr = optimizer.param_groups[0]["lr"]

    class FakeScaler:
        def __init__(self, *, skip_step: bool):
            self._skip_step = skip_step
            self._scale = 1024.0

        def get_scale(self):
            return self._scale

        def step(self, inner_optimizer):
            if not self._skip_step:
                inner_optimizer.step()

        def update(self):
            if self._skip_step:
                self._scale /= 2.0

    completed_steps = 0

    if _apply_optimizer_step(optimizer, FakeScaler(skip_step=True)):
        scheduler.step()
        completed_steps += 1

    assert completed_steps == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)

    if _apply_optimizer_step(optimizer, FakeScaler(skip_step=False)):
        scheduler.step()
        completed_steps += 1

    assert completed_steps == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        compute_lr(
            1,
            warmup_steps=5,
            max_steps=20,
            max_lr=1.0,
            min_lr=0.1,
            schedule="wsd",
            decay_ratio=0.1,
        )
    )


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


def test_mid_epoch_resume_matches_uninterrupted_training_state(tmp_path: Path):
    full_config = _build_tiny_project_config(tmp_path, runs_dir=tmp_path / "runs-full")
    run_train(full_config, device=torch.device("cpu"))

    full_step1_ckpt = _single_checkpoint(Path(full_config.training.runs_dir), "*step_0000001.pt")
    full_final_ckpt = _single_checkpoint(Path(full_config.training.runs_dir), "*step_0000003.pt")

    resume_config = _build_tiny_project_config(tmp_path, runs_dir=tmp_path / "runs-resume")
    run_train(
        resume_config,
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
    assert full_state["scheduler"] == resumed_state["scheduler"]
    _assert_optimizer_states_equal(full_state["optimizer"], resumed_state["optimizer"])
    for key, tensor in full_state["model"].items():
        torch.testing.assert_close(tensor, resumed_state["model"][key])
