"""Tests for small testable helpers inside the SFT trainer.

The training loop itself is integration-tested via smoke runs, but its
guard helpers (NaN-loss detection, LR schedules) are pure functions that
deserve fast unit tests so a refactor cannot regress them silently.
"""

from __future__ import annotations

import math

import torch

from src.post_training.sft.trainer import (
    _cosine_lr,
    _draw_pretraining_window,
    _is_nonfinite,
    _should_stop_early,
    _wsd_lr,
    _wt103_should_stop,
)


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


# ── Early stopping ──────────────────────────────────────────────────────────

def test_should_stop_early_disabled_when_patience_zero():
    """Patience = 0 disables early stopping entirely — never returns True
    no matter how many bad evals. SFT.md §3.4 makes early stopping
    optional (disable for short ablation runs)."""
    assert not _should_stop_early(bad_evals=100, patience=0)


def test_should_stop_early_triggers_at_patience_threshold():
    assert _should_stop_early(bad_evals=5, patience=5)
    assert _should_stop_early(bad_evals=10, patience=5)


def test_should_stop_early_below_threshold():
    assert not _should_stop_early(bad_evals=0, patience=5)
    assert not _should_stop_early(bad_evals=4, patience=5)


# ── Wikitext-103 catastrophic-forgetting tripwire ───────────────────────────

def test_wt103_should_stop_disabled_when_threshold_zero():
    """threshold_pct=0 disables the tripwire — useful for baseline runs."""
    assert not _wt103_should_stop(baseline_ppl=10.0, current_ppl=1000.0, threshold_pct=0.0)


def test_wt103_should_stop_triggers_above_threshold():
    """Current PPL more than threshold% above baseline → trigger."""
    # 15% above 10.0 = 11.5; threshold 10% → 11.0; 11.5 > 11.0
    assert _wt103_should_stop(baseline_ppl=10.0, current_ppl=11.5, threshold_pct=10.0)


def test_wt103_should_stop_at_exact_threshold():
    """Exact threshold should trigger (>= comparison) — the SFT.md
    operating-bound 'rising >10%' is treated as a soft inequality so the
    tripwire fires before scores degrade further."""
    assert _wt103_should_stop(baseline_ppl=10.0, current_ppl=11.0, threshold_pct=10.0)


def test_wt103_should_stop_below_threshold_passes():
    """Within tolerance — the model is still healthy."""
    assert not _wt103_should_stop(baseline_ppl=10.0, current_ppl=10.5, threshold_pct=10.0)


def test_wt103_should_stop_handles_better_ppl():
    """If PPL goes DOWN (model improved on WT-103) the tripwire never fires."""
    assert not _wt103_should_stop(baseline_ppl=10.0, current_ppl=8.0, threshold_pct=10.0)


# ── Pretraining-mix window sampler (CF mitigation #1, VL07 slide 25) ────────

def test_draw_pretraining_window_returns_correct_shapes():
    """Drawn windows are (B, T) inputs and (B, T) targets, shifted by one."""
    import numpy as np
    # Synthetic .bin in-memory: token ids 0..999.
    arr = np.arange(1000, dtype=np.uint16)
    inputs, targets = _draw_pretraining_window(
        arr, context_length=8, batch_size=4, rng=np.random.default_rng(0),
    )
    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    # Targets are inputs shifted by one — VL04 next-token semantics.
    # For each row, targets[t] should equal inputs[t]+1 (because data is sequential).
    assert (targets == inputs + 1).all()


def test_draw_pretraining_window_is_deterministic_per_rng():
    """Same RNG seed → same sampled positions. Required for reproducibility."""
    import numpy as np
    arr = np.arange(1000, dtype=np.uint16)
    a_inputs, _ = _draw_pretraining_window(arr, 8, 4, np.random.default_rng(42))
    b_inputs, _ = _draw_pretraining_window(arr, 8, 4, np.random.default_rng(42))
    assert (a_inputs == b_inputs).all()


def test_draw_pretraining_window_respects_array_bounds():
    """Window starts must leave room for context_length+1 tokens — the
    last valid start is len(arr) - context_length - 1."""
    import numpy as np
    arr = np.arange(20, dtype=np.uint16)
    # context_length=10 → max start = 20 - 10 - 1 = 9
    for _ in range(50):
        inputs, targets = _draw_pretraining_window(
            arr, context_length=10, batch_size=2, rng=np.random.default_rng(),
        )
        assert inputs.shape == (2, 10)
        # Largest target id must fit in the array.
        assert int(targets.max()) < len(arr)
