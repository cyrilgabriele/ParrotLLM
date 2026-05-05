"""Tests for the DPO loss math (VL08 slide 33).

Reference:
    L_DPO = -log σ(β [(π_θ(y_w) - π_θ(y_l)) - (π_ref(y_w) - π_ref(y_l))])

The whole point of DPO is the asymmetric Bradley-Terry sigmoid behaviour
(VL08 slide 14): negative margin → steep gradient (fast correction);
positive margin → small gradient (no over-fitting). These tests pin
that property + the closed-form values at the edges.
"""

from __future__ import annotations

import math

import torch

from src.post_training.dpo.trainer import dpo_loss, per_sequence_logp


def _make_logp(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# ── Closed-form values ──────────────────────────────────────────────────────

def test_dpo_loss_equals_ln2_at_zero_margin():
    """All four logps equal → margin = 0 → loss = -log σ(0) = log(2)."""
    z = _make_logp([0.0])
    loss, m = dpo_loss(
        policy_chosen_logp=z, policy_rejected_logp=z,
        ref_chosen_logp=z, ref_rejected_logp=z, beta=0.1,
    )
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-5)
    assert m["accuracy"] == 0.0  # margin is exactly 0 → not counted as positive


def test_dpo_loss_decreases_with_positive_margin():
    """Policy strongly prefers chosen → loss approaches 0."""
    pol_c = _make_logp([5.0])
    pol_r = _make_logp([0.0])
    ref_c = _make_logp([0.0])
    ref_r = _make_logp([0.0])
    loss, m = dpo_loss(
        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=1.0,
    )
    # logits = 1.0 * (5 - 0) - (0 - 0) = 5; loss = -log σ(5) ≈ 0.0067
    assert loss.item() < math.log(2.0)
    assert m["accuracy"] == 1.0


def test_dpo_loss_increases_with_negative_margin():
    """Policy prefers rejected (wrong ranking) → large loss, fast gradient."""
    pol_c = _make_logp([0.0])
    pol_r = _make_logp([5.0])
    ref_c = _make_logp([0.0])
    ref_r = _make_logp([0.0])
    loss, m = dpo_loss(
        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=1.0,
    )
    # logits = -5; loss = -log σ(-5) ≈ 5.0067
    assert loss.item() > math.log(2.0)
    assert m["accuracy"] == 0.0


def test_dpo_loss_factors_out_reference():
    """If reference perfectly cancels (ref_c-ref_r == pol_c-pol_r), margin is
    zero — DPO sees no signal to update on."""
    pol_c = _make_logp([3.0]); pol_r = _make_logp([1.0])
    ref_c = _make_logp([3.0]); ref_r = _make_logp([1.0])
    loss, _ = dpo_loss(
        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=0.5,
    )
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-5)


def test_dpo_loss_higher_beta_amplifies_signal():
    """Doubling β doubles the input to log σ and reduces loss faster."""
    pol_c = _make_logp([2.0]); pol_r = _make_logp([0.0])
    ref_c = _make_logp([0.0]); ref_r = _make_logp([0.0])
    loss_lo, _ = dpo_loss(
        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=0.1,
    )
    loss_hi, _ = dpo_loss(
        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=1.0,
    )
    assert loss_hi.item() < loss_lo.item()


def test_dpo_loss_accuracy_per_pair():
    """Per-pair accuracy = fraction with reward_margin > 0."""
    # 3 pairs: pos, neg, pos
    pol_c = _make_logp([5.0, 0.0, 3.0])
    pol_r = _make_logp([0.0, 5.0, 0.0])
    ref_c = _make_logp([0.0, 0.0, 0.0])
    ref_r = _make_logp([0.0, 0.0, 0.0])
    _, m = dpo_loss(
        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=0.1,
    )
    # 2 of 3 have positive margin → 2/3
    assert math.isclose(m["accuracy"], 2.0 / 3.0, rel_tol=1e-6)


# ── per_sequence_logp ───────────────────────────────────────────────────────

def test_per_sequence_logp_masks_prompt_tokens():
    """A masked-out (label=-100) position must contribute 0 to the sum."""
    from src.model import ParrotLLM
    cfg = {"model": {
        "vocab_size": 100, "pad_token_id": 99, "bos_token_id": 0, "eos_token_id": 0,
        "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64,
        "context_length": 16, "bias": False, "dropout": 0.0,
        "rope_theta": 10000.0, "gradient_checkpointing": False,
    }}
    model = ParrotLLM(cfg).eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 90, (2, 8))

    # All-masked → logp sum should be 0.0 (no positions contribute).
    labels_all_masked = torch.full_like(ids, -100)
    with torch.no_grad():
        logp = per_sequence_logp(model, ids, labels_all_masked)
    assert torch.allclose(logp, torch.zeros_like(logp), atol=1e-6)


def test_per_sequence_logp_length_normalize_divides_by_count():
    """When length_normalize=True, output is mean log-prob per supervised token."""
    from src.model import ParrotLLM
    cfg = {"model": {
        "vocab_size": 100, "pad_token_id": 99, "bos_token_id": 0, "eos_token_id": 0,
        "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64,
        "context_length": 16, "bias": False, "dropout": 0.0,
        "rope_theta": 10000.0, "gradient_checkpointing": False,
    }}
    model = ParrotLLM(cfg).eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 90, (1, 8))
    labels = ids.clone()
    labels[:, :3] = -100  # mask first 3 positions; 4 supervised in shifted space

    with torch.no_grad():
        sum_logp = per_sequence_logp(model, ids, labels, length_normalize=False)
        mean_logp = per_sequence_logp(model, ids, labels, length_normalize=True)
    # mean = sum / count (count of supervised tokens after shift)
    n_supervised = int((labels[:, 1:] != -100).sum().item())
    assert math.isclose(
        mean_logp.item(), sum_logp.item() / n_supervised, rel_tol=1e-4,
    )
