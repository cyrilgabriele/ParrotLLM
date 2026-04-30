"""DPO loss and sequence-level log-probability helper.

Mirrors the TA's EX08_DPO.ipynb (cells `76c7429e` and `a2522f40`) literally.
Sequence log-prob is the SUM (not mean) over unmasked tokens — the LN-DPO
variant (Phase 2.1) lives in a separate plan.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def sequence_logprob_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log p(target tokens) per sequence, ignoring -100 positions.

    Args:
        logits: [B, S, V] float tensor.
        labels: [B, S] long tensor with -100 on positions to ignore.

    Returns:
        [B] float tensor of per-sequence summed log-probs.
    """
    log_probs = F.log_softmax(logits, dim=-1)

    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0  # placeholder index for gather

    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).to(log_probs.dtype)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Canonical DPO loss with implicit-reward diagnostics.

    L = -log sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))
    """
    policy_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    advantages = policy_logratios - ref_logratios
    losses = -F.logsigmoid(beta * advantages)

    implicit_reward_chosen = beta * (policy_chosen_logp - ref_chosen_logp)
    implicit_reward_rejected = beta * (policy_rejected_logp - ref_rejected_logp)

    metrics = {
        "policy_logratios": policy_logratios.detach().mean().item(),
        "ref_logratios": ref_logratios.detach().mean().item(),
        "advantages": advantages.detach().mean().item(),
        "implicit_reward_chosen": implicit_reward_chosen.detach().mean().item(),
        "implicit_reward_rejected": implicit_reward_rejected.detach().mean().item(),
        "implicit_reward_margin": (implicit_reward_chosen - implicit_reward_rejected).detach().mean().item(),
    }
    return losses.mean(), metrics
