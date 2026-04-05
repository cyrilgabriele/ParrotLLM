"""Tests for run-local checkpoint storage and retention."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.training.trainer import CheckpointManager, resolve_checkpoint_dir


def _build_state():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"model": {"d_model": 4}, "training": {"learning_rate": 1e-3}}
    return model, optimizer, config


def test_resolve_checkpoint_dir_under_run_dir(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_001"
    run_dir.mkdir(parents=True)

    checkpoint_dir = resolve_checkpoint_dir(str(run_dir), "checkpoints")

    assert checkpoint_dir == str(run_dir / "checkpoints")
    assert Path(checkpoint_dir).is_dir()


def test_resolve_checkpoint_dir_rejects_escape(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_001"
    run_dir.mkdir(parents=True)

    with pytest.raises(ValueError):
        resolve_checkpoint_dir(str(run_dir), "../outside")


def test_checkpoint_manager_keeps_only_last_n_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(str(checkpoint_dir), keep_last=3, keep_best=0)
    model, optimizer, config = _build_state()

    for step in range(1, 6):
        manager.save_last(
            model,
            optimizer,
            config,
            step=step,
            epoch=0,
            scaler=None,
        )

    saved = sorted(p.name for p in checkpoint_dir.glob("*.pt"))
    assert saved == [
        "last_epoch_0000_step_0000003.pt",
        "last_epoch_0000_step_0000004.pt",
        "last_epoch_0000_step_0000005.pt",
    ]


def test_checkpoint_manager_keeps_only_best_n_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(str(checkpoint_dir), keep_last=0, keep_best=2)
    model, optimizer, config = _build_state()

    losses = [
        (1, 5.0),
        (2, 4.0),
        (3, 4.5),
        (4, 3.0),
    ]
    for step, loss in losses:
        manager.maybe_save_best(
            model,
            optimizer,
            config,
            step=step,
            epoch=0,
            scaler=None,
            val_loss=loss,
        )

    saved = sorted(p.name for p in checkpoint_dir.glob("*.pt"))
    assert saved == [
        "best_loss_3p0000_epoch_0000_step_0000004.pt",
        "best_loss_4p0000_epoch_0000_step_0000002.pt",
    ]
    assert manager.best_path is not None
    assert manager.best_path.endswith("best_loss_3p0000_epoch_0000_step_0000004.pt")


def test_checkpoint_manager_replaces_duplicate_last_step(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(str(checkpoint_dir), keep_last=2, keep_best=0)
    model, optimizer, config = _build_state()

    manager.save_last(model, optimizer, config, step=10, epoch=0, scaler=None)
    manager.save_last(model, optimizer, config, step=10, epoch=0, scaler=None)

    saved = list(checkpoint_dir.glob("*.pt"))
    assert len(saved) == 1
