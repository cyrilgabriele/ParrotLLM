"""Numerics for sequence_logprob_from_labels and dpo_loss must match EX08_DPO.ipynb.

Source: EX08_DPO.ipynb cells `76c7429e` (sequence_logprob_from_labels)
and `a2522f40` (dpo_loss).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.posttraining.dpo.loss import dpo_loss, sequence_logprob_from_labels


def test_sequence_logprob_sums_only_unmasked_tokens() -> None:
    # 1 sequence, 4 positions, vocab=3.
    # The first token is masked (-100); the rest carry log-probs we'll check.
    torch.manual_seed(0)
    logits = torch.tensor([
        [
            [0.0, 0.0, 0.0],   # ignored due to -100 label
            [1.0, 2.0, 3.0],   # target=1 -> log_softmax[1]
            [4.0, 5.0, 6.0],   # target=2 -> log_softmax[2]
            [0.0, 1.0, 0.0],   # target=0 -> log_softmax[0]
        ]
    ])
    labels = torch.tensor([[-100, 1, 2, 0]])
    log_probs = F.log_softmax(logits, dim=-1)
    expected = (log_probs[0, 1, 1] + log_probs[0, 2, 2] + log_probs[0, 3, 0]).item()
    got = sequence_logprob_from_labels(logits, labels).tolist()[0]
    assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_sequence_logprob_returns_zero_when_all_masked() -> None:
    logits = torch.zeros((1, 3, 4))
    labels = torch.full((1, 3), -100)
    got = sequence_logprob_from_labels(logits, labels)
    assert got.shape == (1,)
    assert got.item() == 0.0


def test_dpo_loss_matches_closed_form_at_known_inputs() -> None:
    # Closed form: L = -log sigma(beta * advantage)
    # advantage = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    pi_c = torch.tensor([2.0, 1.5])
    pi_r = torch.tensor([1.0, 1.0])
    ref_c = torch.tensor([1.5, 1.4])
    ref_r = torch.tensor([1.4, 1.0])
    beta = 0.1
    expected_advantage = ((pi_c - pi_r) - (ref_c - ref_r)).mean().item()
    expected_loss = (-F.logsigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))).mean().item()
    loss, metrics = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)
    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["advantages"], expected_advantage, rel_tol=1e-6, abs_tol=1e-6)
    expected_margin = (
        beta * (pi_c - ref_c) - beta * (pi_r - ref_r)
    ).mean().item()
    assert math.isclose(metrics["implicit_reward_margin"], expected_margin, rel_tol=1e-6, abs_tol=1e-6)


def test_dpo_loss_returns_all_required_metrics() -> None:
    pi_c = torch.tensor([1.0])
    pi_r = torch.tensor([0.0])
    ref_c = torch.tensor([0.5])
    ref_r = torch.tensor([0.5])
    _, metrics = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    # Per the TA notebook: loss, policy_logratios, ref_logratios, advantages,
    # implicit_reward_chosen, implicit_reward_rejected, implicit_reward_margin.
    required = {
        "policy_logratios", "ref_logratios", "advantages",
        "implicit_reward_chosen", "implicit_reward_rejected",
        "implicit_reward_margin",
    }
    assert required <= set(metrics)


def test_dpo_loss_decreases_when_policy_preferred_chosen() -> None:
    # If the policy raises chosen logp relative to ref, the loss should drop.
    pi_c_a = torch.tensor([1.0]); pi_r_a = torch.tensor([1.0])
    pi_c_b = torch.tensor([2.0]); pi_r_b = torch.tensor([1.0])
    ref_c = torch.tensor([1.0]); ref_r = torch.tensor([1.0])
    loss_a, _ = dpo_loss(pi_c_a, pi_r_a, ref_c, ref_r, beta=0.1)
    loss_b, _ = dpo_loss(pi_c_b, pi_r_b, ref_c, ref_r, beta=0.1)
    assert loss_b.item() < loss_a.item()
