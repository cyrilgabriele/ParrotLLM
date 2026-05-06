"""DPO loss and sequence-level log-probability helper.

Mirrors the TA's EX08_DPO.ipynb (cells `76c7429e` and `a2522f40`) literally
for the default (sum-of-logp) path. The ``length_normalize`` flag enables the
LN-DPO variant motivated by the cross-source length-asymmetry audit at
``docs/superpowers/notes/2026-05-06-dpo-pair-length-audit.md``: per-pair
gradient signal becomes the MEAN log-prob over unmasked tokens, so a 14-token
HellaSwag pair no longer dominates a 1.6-token WinoGrande pair by ~9x.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def sequence_logprob_from_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Per-sequence log p(target tokens), ignoring -100 positions.

    Args:
        logits: [B, S, V] float tensor.
        labels: [B, S] long tensor with -100 on positions to ignore.
        length_normalize: when False (default, legacy behavior), return the
            SUM of per-token log-probs. When True, return the MEAN over
            unmasked positions per row — equalising per-pair gradient
            magnitude across sources of differing response length. Rows with
            zero unmasked positions return 0.0 (divisor clamped to >= 1).

    Returns:
        [B] float tensor of per-sequence (summed or mean) log-probs.
    """
    # log p(t) = logits[t] - logsumexp(logits) — avoids materialising the
    # [B, S, V] log_softmax tensor (~800 MB at our shapes) and the matching
    # autograd activation. Mathematically identical to F.log_softmax + gather.
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0  # placeholder index for gather

    token_logits = logits.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    log_partition = torch.logsumexp(logits, dim=-1)
    token_log_probs = token_logits - log_partition
    mask = (labels != -100).to(token_log_probs.dtype)
    summed = (token_log_probs * mask).sum(dim=-1)
    if not length_normalize:
        return summed
    # Per-row count of supervised tokens, clamped to >= 1 so all-masked rows
    # do not divide by zero. (The summed log-prob is already 0 for those.)
    counts = mask.sum(dim=-1).clamp(min=1.0)
    return summed / counts


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
