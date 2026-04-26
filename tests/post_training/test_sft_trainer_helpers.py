"""Tests for small testable helpers inside the SFT trainer.

The training loop itself is integration-tested via smoke runs, but its
guard helpers (NaN-loss detection, LR schedules) are pure functions that
deserve fast unit tests so a refactor cannot regress them silently.
"""

from __future__ import annotations

import math

import torch

from src.post_training.sft.trainer import _cosine_lr, _is_nonfinite, _wsd_lr


# ── NaN-loss guard ───────────────────────────────────────────────────────────

def test_is_nonfinite_detects_nan():
    """A NaN loss must trigger the skip path — otherwise it propagates into
    the optimiser and corrupts AdamW state for the rest of the run."""
    assert _is_nonfinite(torch.tensor(float("nan")))


def test_is_nonfinite_detects_positive_inf():
    assert _is_nonfinite(torch.tensor(float("inf")))


def test_is_nonfinite_detects_negative_inf():
    assert _is_nonfinite(torch.tensor(float("-inf")))


def test_is_nonfinite_passes_through_normal_loss():
    assert not _is_nonfinite(torch.tensor(1.234))


# ── LR schedules (smoke + boundary conditions) ──────────────────────────────

def test_cosine_lr_warmup_is_linear_to_peak():
    peak = 2e-5
    floor = 2e-6
    warmup = 100
    total = 1000
    # At step 0 we should be just above zero, at step warmup-1 ≈ peak.
    assert _cosine_lr(0, warmup, total, peak, floor) == peak * (1 / warmup)
    assert math.isclose(_cosine_lr(warmup - 1, warmup, total, peak, floor), peak)


def test_cosine_lr_floors_at_min_after_total():
    peak, floor, warmup, total = 2e-5, 2e-6, 100, 1000
    assert _cosine_lr(total, warmup, total, peak, floor) == floor
    assert _cosine_lr(total + 100, warmup, total, peak, floor) == floor


def test_wsd_lr_holds_peak_during_stable_phase():
    peak, floor = 1.0, 0.0
    warmup, total = 10, 100
    # decay_ratio default = 0.1 → decay_steps = 10, decay_start = 90.
    # Mid-stable-phase: any step in [warmup, decay_start) returns peak.
    assert _wsd_lr(50, warmup, total, peak, floor) == peak
    assert _wsd_lr(89, warmup, total, peak, floor) == peak


def test_wsd_lr_decays_linearly_to_floor():
    peak, floor = 1.0, 0.0
    warmup, total = 10, 100
    # decay_start = 90, decay_steps = 10 → at step 95 we are halfway through decay.
    assert math.isclose(_wsd_lr(95, warmup, total, peak, floor), 0.5)
    assert _wsd_lr(total, warmup, total, peak, floor) == floor
