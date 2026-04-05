"""Tests for find_latest_checkpoint and its helper utilities."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.training.trainer import (
    _is_checkpoint_candidate,
    _parse_step_from_filename,
    _validate_checkpoint,
    find_latest_checkpoint,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_checkpoint(path: str | Path, step: int = 1) -> None:
    """Write a minimal valid checkpoint to *path*."""
    torch.save({"model": {}, "optimizer": {}, "step": step, "config": {}}, str(path))


def _make_corrupt_checkpoint(path: str | Path) -> None:
    """Write a file that is not a valid PyTorch checkpoint."""
    Path(path).write_bytes(b"this is not a pytorch file")


def _run_dir(runs_root: Path, name: str) -> Path:
    d = runs_root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_dir(run_dir: Path, subdir: str = "checkpoints") -> Path:
    d = run_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── _parse_step_from_filename ─────────────────────────────────────────────────

class TestParseStepFromFilename:
    def test_last_format(self):
        assert _parse_step_from_filename("last_epoch_0001_step_0005000.pt") == 5000

    def test_best_format(self):
        assert _parse_step_from_filename(
            "best_loss_2p4500_epoch_0002_step_0010000.pt"
        ) == 10000

    def test_leading_zeros_stripped(self):
        assert _parse_step_from_filename("last_epoch_0000_step_0000001.pt") == 1

    def test_legacy_format_no_extension(self):
        assert _parse_step_from_filename("00_epoch_1000_step") == 1000

    def test_legacy_format_large_step(self):
        assert _parse_step_from_filename("02_epoch_50000_step") == 50000

    def test_no_step_returns_none(self):
        assert _parse_step_from_filename("my_custom_checkpoint.pt") is None

    def test_empty_string_returns_none(self):
        assert _parse_step_from_filename("") is None


# ── _is_checkpoint_candidate ─────────────────────────────────────────────────

class TestIsCheckpointCandidate:
    def test_pt_extension_accepted(self):
        assert _is_checkpoint_candidate("last_epoch_0001_step_0005000.pt") is True

    def test_legacy_no_extension_accepted(self):
        assert _is_checkpoint_candidate("00_epoch_1000_step") is True

    def test_json_file_rejected(self):
        assert _is_checkpoint_candidate("config.json") is False

    def test_log_file_rejected(self):
        assert _is_checkpoint_candidate("train.log") is False

    def test_pdf_rejected(self):
        assert _is_checkpoint_candidate("training_plots.pdf") is False

    def test_jsonl_rejected(self):
        assert _is_checkpoint_candidate("metrics.jsonl") is False


# ── _validate_checkpoint ──────────────────────────────────────────────────────

class TestValidateCheckpoint:
    def test_valid_checkpoint_returns_true(self, tmp_path: Path):
        p = tmp_path / "ckpt.pt"
        _make_checkpoint(p, step=42)
        ok, reason = _validate_checkpoint(str(p))
        assert ok is True
        assert reason == ""

    def test_corrupt_file_returns_false_with_reason(self, tmp_path: Path):
        p = tmp_path / "bad.pt"
        _make_corrupt_checkpoint(p)
        ok, reason = _validate_checkpoint(str(p))
        assert ok is False
        assert reason  # non-empty error message

    def test_missing_file_returns_false_with_reason(self, tmp_path: Path):
        ok, reason = _validate_checkpoint(str(tmp_path / "nonexistent.pt"))
        assert ok is False
        assert reason

    def test_missing_model_key_returns_false(self, tmp_path: Path):
        p = tmp_path / "ckpt.pt"
        torch.save({"step": 10, "optimizer": {}}, str(p))
        ok, reason = _validate_checkpoint(str(p))
        assert ok is False
        assert "model" in reason

    def test_missing_step_key_returns_false(self, tmp_path: Path):
        p = tmp_path / "ckpt.pt"
        torch.save({"model": {}, "optimizer": {}}, str(p))
        ok, reason = _validate_checkpoint(str(p))
        assert ok is False
        assert "step" in reason

    def test_non_dict_payload_returns_false(self, tmp_path: Path):
        p = tmp_path / "ckpt.pt"
        torch.save([1, 2, 3], str(p))
        ok, reason = _validate_checkpoint(str(p))
        assert ok is False
        assert reason


# ── find_latest_checkpoint ────────────────────────────────────────────────────

class TestFindLatestCheckpointHappyPaths:
    def test_returns_highest_step_checkpoint(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        _make_checkpoint(ckpt_dir / "last_epoch_0000_step_0005000.pt", step=5000)
        _make_checkpoint(ckpt_dir / "last_epoch_0000_step_0010000.pt", step=10000)
        _make_checkpoint(ckpt_dir / "last_epoch_0001_step_0015000.pt", step=15000)

        result = find_latest_checkpoint(str(tmp_path))
        assert result.endswith("last_epoch_0001_step_0015000.pt")

    def test_auto_selects_most_recent_run(self, tmp_path: Path):
        old_run = _run_dir(tmp_path, "run_20260101_000000")
        new_run = _run_dir(tmp_path, "run_20260405_120000")
        _make_checkpoint(_checkpoint_dir(old_run) / "last_epoch_0000_step_0099999.pt", step=99999)
        _make_checkpoint(_checkpoint_dir(new_run) / "last_epoch_0000_step_0001000.pt", step=1000)

        result = find_latest_checkpoint(str(tmp_path))
        # Must choose from the newer run, even though old run has a higher step
        assert "run_20260405_120000" in result

    def test_falls_back_to_older_run_when_newest_has_no_checkpoints(self, tmp_path: Path):
        old_run = _run_dir(tmp_path, "run_20260101_000000")
        new_run = _run_dir(tmp_path, "run_20260405_120000")
        # New run: exists but no checkpoints directory (crashed before first save)
        # Old run: has a valid checkpoint
        _make_checkpoint(_checkpoint_dir(old_run) / "last_epoch_0000_step_0005000.pt", step=5000)

        result = find_latest_checkpoint(str(tmp_path))
        assert "run_20260101_000000" in result
        assert result.endswith("last_epoch_0000_step_0005000.pt")

    def test_falls_back_across_multiple_empty_runs(self, tmp_path: Path):
        for name in ["run_20260405_120000", "run_20260405_110000", "run_20260101_000000"]:
            _run_dir(tmp_path, name)
        # Only the oldest has checkpoints
        old_run = tmp_path / "run_20260101_000000"
        _make_checkpoint(_checkpoint_dir(old_run) / "last_epoch_0000_step_0001000.pt", step=1000)

        result = find_latest_checkpoint(str(tmp_path))
        assert "run_20260101_000000" in result

    def test_falls_back_when_newest_checkpoint_dir_is_empty(self, tmp_path: Path):
        new_run = _run_dir(tmp_path, "run_20260405_120000")
        old_run = _run_dir(tmp_path, "run_20260101_000000")
        _checkpoint_dir(new_run)  # empty checkpoint dir
        _make_checkpoint(_checkpoint_dir(old_run) / "last_epoch_0000_step_0003000.pt", step=3000)

        result = find_latest_checkpoint(str(tmp_path))
        assert "run_20260101_000000" in result

    def test_explicit_run_dir_overrides_auto_detection(self, tmp_path: Path):
        run_a = _run_dir(tmp_path, "run_20260405_120000")
        run_b = _run_dir(tmp_path, "run_20260405_130000")
        _make_checkpoint(_checkpoint_dir(run_a) / "last_epoch_0000_step_0005000.pt", step=5000)
        _make_checkpoint(_checkpoint_dir(run_b) / "last_epoch_0000_step_0001000.pt", step=1000)

        # Explicitly ask for run_a even though run_b is newer
        result = find_latest_checkpoint(str(tmp_path), run_dir=str(run_a))
        assert "run_20260405_120000" in result
        assert result.endswith("last_epoch_0000_step_0005000.pt")

    def test_prefers_last_over_best_at_same_step(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        _make_checkpoint(ckpt_dir / "best_loss_2p3000_epoch_0001_step_0010000.pt", step=10000)
        _make_checkpoint(ckpt_dir / "last_epoch_0001_step_0010000.pt", step=10000)

        result = find_latest_checkpoint(str(tmp_path))
        assert os.path.basename(result).startswith("last_")

    def test_falls_back_to_best_when_no_last_checkpoints(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        _make_checkpoint(ckpt_dir / "best_loss_2p3000_epoch_0001_step_0010000.pt", step=10000)

        result = find_latest_checkpoint(str(tmp_path))
        assert "best_loss" in os.path.basename(result)

    def test_custom_checkpoint_subdir(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        custom_dir = run / "my_ckpts"
        custom_dir.mkdir()
        _make_checkpoint(custom_dir / "last_epoch_0000_step_0001000.pt", step=1000)

        result = find_latest_checkpoint(
            str(tmp_path), checkpoint_subdir="my_ckpts"
        )
        assert "my_ckpts" in result

    def test_returns_absolute_path(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        _make_checkpoint(_checkpoint_dir(run) / "last_epoch_0000_step_0001000.pt", step=1000)

        result = find_latest_checkpoint(str(tmp_path))
        assert os.path.isabs(result)


class TestFindLatestCheckpointValidation:
    def test_skips_corrupt_file_and_returns_next_best(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        # Higher step but corrupted
        _make_corrupt_checkpoint(ckpt_dir / "last_epoch_0001_step_0020000.pt")
        # Lower step but valid
        _make_checkpoint(ckpt_dir / "last_epoch_0000_step_0010000.pt", step=10000)

        result = find_latest_checkpoint(str(tmp_path), validate=True)
        assert result.endswith("last_epoch_0000_step_0010000.pt")

    def test_skips_multiple_corrupt_files_in_order(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        _make_corrupt_checkpoint(ckpt_dir / "last_epoch_0002_step_0030000.pt")
        _make_corrupt_checkpoint(ckpt_dir / "last_epoch_0001_step_0020000.pt")
        _make_checkpoint(ckpt_dir / "last_epoch_0000_step_0010000.pt", step=10000)

        result = find_latest_checkpoint(str(tmp_path), validate=True)
        assert result.endswith("last_epoch_0000_step_0010000.pt")

    def test_raises_when_all_checkpoints_corrupt(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        _make_corrupt_checkpoint(ckpt_dir / "last_epoch_0000_step_0010000.pt")
        _make_corrupt_checkpoint(ckpt_dir / "last_epoch_0001_step_0020000.pt")

        with pytest.raises(FileNotFoundError, match="No valid checkpoint found"):
            find_latest_checkpoint(str(tmp_path), validate=True)

    def test_validate_false_accepts_corrupt_file(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        _make_corrupt_checkpoint(ckpt_dir / "last_epoch_0000_step_0010000.pt")

        # Should not raise when validation is disabled
        result = find_latest_checkpoint(str(tmp_path), validate=False)
        assert result.endswith("last_epoch_0000_step_0010000.pt")


class TestFindLatestCheckpointErrorCases:
    def test_missing_runs_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Runs directory does not exist"):
            find_latest_checkpoint(str(tmp_path / "nonexistent"))

    def test_empty_runs_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No run_\\* directories found"):
            find_latest_checkpoint(str(tmp_path))

    def test_runs_dir_with_no_run_prefix_dirs_raises(self, tmp_path: Path):
        (tmp_path / "other_dir").mkdir()
        with pytest.raises(FileNotFoundError, match="No run_\\* directories found"):
            find_latest_checkpoint(str(tmp_path))

    def test_specified_run_dir_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            find_latest_checkpoint(str(tmp_path), run_dir=str(tmp_path / "nonexistent_run"))

    def test_missing_checkpoint_subdir_raises(self, tmp_path: Path):
        _run_dir(tmp_path, "run_20260405_120000")
        # No checkpoints subdir created — all runs exhausted

        with pytest.raises(FileNotFoundError, match="No valid checkpoint found"):
            find_latest_checkpoint(str(tmp_path))

    def test_missing_checkpoint_subdir_explicit_run_raises_targeted_error(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        # No checkpoints subdir — explicit run_dir should give targeted message

        with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
            find_latest_checkpoint(str(tmp_path), run_dir=str(run))

    def test_empty_checkpoint_dir_raises(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        _checkpoint_dir(run)  # create empty dir

        with pytest.raises(FileNotFoundError, match="No valid checkpoint found"):
            find_latest_checkpoint(str(tmp_path))

    def test_ignores_non_pt_files(self, tmp_path: Path):
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        (ckpt_dir / "config.json").write_text("{}")
        (ckpt_dir / "train.log").write_text("log output")

        with pytest.raises(FileNotFoundError, match="No valid checkpoint found"):
            find_latest_checkpoint(str(tmp_path))

    def test_finds_legacy_checkpoints_in_run_root(self, tmp_path: Path):
        """Old runs stored checkpoints directly in the run root without a subdir."""
        run = _run_dir(tmp_path, "run_20260320_145519")
        # Legacy format: no .pt extension, stored at run root
        _make_checkpoint(run / "00_epoch_250_step", step=250)
        _make_checkpoint(run / "00_epoch_500_step", step=500)

        result = find_latest_checkpoint(str(tmp_path))
        assert os.path.basename(result) == "00_epoch_500_step"

    def test_prefers_newer_checkpoint_subdir_over_legacy_run_root(self, tmp_path: Path):
        """If both layouts coexist, pick the highest step regardless of location."""
        run = _run_dir(tmp_path, "run_20260405_120000")
        _make_checkpoint(run / "00_epoch_1000_step", step=1000)
        _make_checkpoint(_checkpoint_dir(run) / "last_epoch_0001_step_0002000.pt", step=2000)

        result = find_latest_checkpoint(str(tmp_path))
        assert result.endswith("last_epoch_0001_step_0002000.pt")

    def test_non_standard_filenames_use_mtime_fallback(self, tmp_path: Path):
        """Checkpoints without step numbers in their name are ranked by mtime."""
        run = _run_dir(tmp_path, "run_20260405_120000")
        ckpt_dir = _checkpoint_dir(run)
        p1 = ckpt_dir / "checkpoint_a.pt"
        p2 = ckpt_dir / "checkpoint_b.pt"
        _make_checkpoint(p1, step=1)
        _make_checkpoint(p2, step=2)
        # Touch p2 to make it newer
        os.utime(p1, (1000.0, 1000.0))
        os.utime(p2, (2000.0, 2000.0))

        result = find_latest_checkpoint(str(tmp_path))
        assert result.endswith("checkpoint_b.pt")
