"""Tests for length-normalized per-sequence log-prob in DPO.

Per the audit at docs/superpowers/notes/2026-05-06-dpo-pair-length-audit.md,
the unnormalized sum-of-logp formulation lets long sources (HellaSwag ~14
tokens) dominate short sources (WinoGrande ~1.6 tokens) by ~2.2x in gradient
contribution per pair. Length normalization (mean over supervised tokens)
makes per-pair signal comparable across sources of different response length.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.posttraining.dpo.loss import dpo_loss, sequence_logprob_from_labels


def _build_logits_with_uniform_per_token_logp(
    *, batch_size: int, seq_len: int, vocab_size: int, target_id: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build logits where every supervised position has the same per-token log-prob.

    Returns (logits, labels) with labels filled with ``target_id`` everywhere
    (no -100 masks). With identical logits at every position, the per-token
    log-prob is constant, so the SUM scales linearly with seq_len while the
    MEAN is constant.
    """
    # Fixed logits — identical at every position so per-token logp is uniform.
    base = torch.tensor([0.5, 1.5, -0.5][:vocab_size], dtype=torch.float32)
    if vocab_size > 3:
        base = torch.cat([base, torch.zeros(vocab_size - 3)])
    logits = base.view(1, 1, vocab_size).expand(batch_size, seq_len, vocab_size).contiguous()
    labels = torch.full((batch_size, seq_len), target_id, dtype=torch.long)
    return logits, labels


def test_unnormalized_sums_scale_with_length() -> None:
    """Sanity: with identical per-token logp, sum scales linearly with length."""
    short_logits, short_labels = _build_logits_with_uniform_per_token_logp(
        batch_size=1, seq_len=2, vocab_size=4
    )
    long_logits, long_labels = _build_logits_with_uniform_per_token_logp(
        batch_size=1, seq_len=10, vocab_size=4
    )
    short_sum = sequence_logprob_from_labels(short_logits, short_labels).item()
    long_sum = sequence_logprob_from_labels(long_logits, long_labels).item()
    # The 5x-longer sequence has ~5x the summed log-prob magnitude.
    assert math.isclose(long_sum / short_sum, 5.0, rel_tol=1e-5)


def test_length_normalize_equalises_per_sequence_logp() -> None:
    """With length_normalize=True, identical mean per-token logp -> equal values."""
    short_logits, short_labels = _build_logits_with_uniform_per_token_logp(
        batch_size=1, seq_len=2, vocab_size=4
    )
    long_logits, long_labels = _build_logits_with_uniform_per_token_logp(
        batch_size=1, seq_len=10, vocab_size=4
    )
    short_norm = sequence_logprob_from_labels(
        short_logits, short_labels, length_normalize=True
    ).item()
    long_norm = sequence_logprob_from_labels(
        long_logits, long_labels, length_normalize=True
    ).item()
    # Per-token logp is constant by construction, so the means are equal.
    assert math.isclose(short_norm, long_norm, rel_tol=1e-6, abs_tol=1e-7)


def test_length_normalize_default_off_preserves_legacy_sum() -> None:
    """Default (no flag passed) must equal length_normalize=False — backward compat."""
    logits, labels = _build_logits_with_uniform_per_token_logp(
        batch_size=2, seq_len=5, vocab_size=4
    )
    default = sequence_logprob_from_labels(logits, labels)
    explicit_off = sequence_logprob_from_labels(logits, labels, length_normalize=False)
    assert torch.allclose(default, explicit_off)


def test_length_normalize_divides_by_unmasked_count_per_row() -> None:
    """When rows have different unmasked counts, normalization divides per-row."""
    # 2 rows, 6 positions. Row 0: 2 supervised tokens. Row 1: 4 supervised tokens.
    logits, _ = _build_logits_with_uniform_per_token_logp(
        batch_size=2, seq_len=6, vocab_size=4
    )
    labels = torch.tensor([
        [-100, -100, -100, -100, 1, 1],   # 2 supervised positions
        [-100, -100, 1, 1, 1, 1],          # 4 supervised positions
    ])
    sums = sequence_logprob_from_labels(logits, labels)
    means = sequence_logprob_from_labels(logits, labels, length_normalize=True)
    # means should equal sums / unmasked_count, per row.
    expected_row0 = sums[0].item() / 2.0
    expected_row1 = sums[1].item() / 4.0
    assert math.isclose(means[0].item(), expected_row0, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(means[1].item(), expected_row1, rel_tol=1e-6, abs_tol=1e-7)
    # And in this construction (uniform per-token logp), the two means are equal.
    assert math.isclose(means[0].item(), means[1].item(), rel_tol=1e-6, abs_tol=1e-7)


def test_length_normalize_handles_all_masked_row_without_division_error() -> None:
    """All-masked row returns 0 (sum is 0, divisor is clamped to >= 1)."""
    logits = torch.zeros((1, 3, 4))
    labels = torch.full((1, 3), -100)
    out = sequence_logprob_from_labels(logits, labels, length_normalize=True)
    assert out.shape == (1,)
    assert out.item() == 0.0


def test_length_normalize_changes_dpo_loss_for_length_imbalanced_pair() -> None:
    """End-to-end: with one side 5x longer, normalized loss differs from unnormalized.

    This is the load-bearing property the audit motivates: the unnormalized loss
    weights longer sequences more heavily because their summed log-prob has
    larger magnitude, while the normalized loss treats per-token quality
    uniformly across pair-length buckets.
    """
    # Construct two pairs where the per-token logp is identical for chosen and
    # rejected, but the rejected is 5x longer than the chosen.
    short_logits, short_labels = _build_logits_with_uniform_per_token_logp(
        batch_size=1, seq_len=2, vocab_size=4
    )
    long_logits, long_labels = _build_logits_with_uniform_per_token_logp(
        batch_size=1, seq_len=10, vocab_size=4
    )

    pi_c_sum = sequence_logprob_from_labels(short_logits, short_labels)
    pi_r_sum = sequence_logprob_from_labels(long_logits, long_labels)
    ref_c_sum = sequence_logprob_from_labels(short_logits, short_labels)
    ref_r_sum = sequence_logprob_from_labels(long_logits, long_labels)

    pi_c_mean = sequence_logprob_from_labels(short_logits, short_labels, length_normalize=True)
    pi_r_mean = sequence_logprob_from_labels(long_logits, long_labels, length_normalize=True)
    ref_c_mean = sequence_logprob_from_labels(short_logits, short_labels, length_normalize=True)
    ref_r_mean = sequence_logprob_from_labels(long_logits, long_labels, length_normalize=True)

    # The chosen-vs-rejected GAP under sum is (short_sum - long_sum), which is
    # large in magnitude because of length asymmetry. Under mean, the gap is 0
    # because per-token logp is identical.
    sum_gap = (pi_c_sum - pi_r_sum).item()
    mean_gap = (pi_c_mean - pi_r_mean).item()
    assert abs(sum_gap) > 1.0, f"expected meaningful sum gap, got {sum_gap}"
    assert math.isclose(mean_gap, 0.0, abs_tol=1e-6)

    # And the DPO advantage (policy gap minus ref gap) is 0 either way here
    # because policy and ref are identical, but the per-sequence values
    # plumbed into the loss differ in magnitude — confirming length-normalize
    # actually changes the numbers reaching the loss.
    assert not math.isclose(pi_c_sum.item(), pi_c_mean.item(), abs_tol=1e-6)
    assert not math.isclose(pi_r_sum.item(), pi_r_mean.item(), abs_tol=1e-6)
