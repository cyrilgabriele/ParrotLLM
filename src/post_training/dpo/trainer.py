"""DPO training loop for ParrotLLM (VL08 implementation).

Mirrors the SFT trainer scaffold (mixed precision, schedule, JSONL
metrics, checkpoint-save schema, WT-103 CF tripwire) but swaps the SFT
masked-CE loss for the DPO contrast.

Two models live in memory:
- POLICY (π_θ): the model being trained, initialised from the SFT
  checkpoint passed via --checkpoint.
- REFERENCE (π_ref): a frozen, eval-mode copy of the same SFT
  checkpoint, used only for log-prob computation. No grad flows through it.

Per-batch: 4 forward passes
    π_θ on chosen      → policy_chosen_logp
    π_θ on rejected    → policy_rejected_logp
    π_ref on chosen    → ref_chosen_logp     (no_grad)
    π_ref on rejected  → ref_rejected_logp   (no_grad)

DPO loss (VL08 slide 33):
    L = -log σ(β [(policy_chosen_logp - policy_rejected_logp)
                 - (ref_chosen_logp - ref_rejected_logp)])

per-sequence log-prob = sum (or mean, if length_normalize_logp) of
log p(token_t | prefix, prev tokens) at supervised positions only.
The -100 mask comes from the DPO collator and matches the SFT
convention exactly.

Vanilla PyTorch — no TRL.
"""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import ProjectConfig
from configs.post_training.dpoConfig import DPOConfig
from src.eval.perplexity import compute_perplexity
from src.logging_utils import make_run_dir
from src.model import ParrotLLM
from src.post_training.dpo.collator import (
    DPOCollator, IGNORE_INDEX, count_supervised_tokens,
)
from src.post_training.dpo.data import build_dpo_datasets
from src.post_training.sft.data import load_decontam_texts
from src.post_training.sft.trainer import (
    _cosine_lr,
    _empty_device_cache,
    _is_recoverable_mps_oom,
    _is_nonfinite,
    _load_wt103_tokens,
    _model_parameters_are_finite,
    _should_stop_early,
    _wsd_lr,
    _wt103_should_stop,
)
from src.utils import build_tokenizer, get_device


log = logging.getLogger("parrotllm.dpo.trainer")


# ── DPO loss math (vanilla PyTorch, slide 33) ───────────────────────────────

def dpo_loss(
    *,
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the DPO loss + diagnostic metrics.

    All inputs are 1-D tensors of shape ``(B,)`` containing per-sequence
    log-prob sums (over response tokens only).

    Returns:
        (loss, metrics) where metrics has the standard DPO diagnostics
        used in the lecture / paper:
          - ``policy_logratios``: π_θ(chosen)/π_θ(rejected) avg
          - ``ref_logratios``:    π_ref(chosen)/π_ref(rejected) avg
          - ``rewards_chosen``:   β·(π_θ(chosen) - π_ref(chosen)) avg
          - ``rewards_rejected``: β·(π_θ(rejected) - π_ref(rejected)) avg
          - ``reward_margin``:    rewards_chosen - rewards_rejected avg
          - ``accuracy``:         fraction of pairs with margin > 0
    """
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    rewards_chosen = beta * (policy_chosen_logp - ref_chosen_logp).detach()
    rewards_rejected = beta * (policy_rejected_logp - ref_rejected_logp).detach()
    reward_margin = rewards_chosen - rewards_rejected
    metrics = {
        "policy_logratios": pi_logratios.mean().item(),
        "ref_logratios": ref_logratios.mean().item(),
        "rewards_chosen": rewards_chosen.mean().item(),
        "rewards_rejected": rewards_rejected.mean().item(),
        "reward_margin": reward_margin.mean().item(),
        "accuracy": (reward_margin > 0).float().mean().item(),
    }
    return loss, metrics


def per_sequence_logp(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalize: bool = False,
    loss_chunk_rows: int = 2048,
) -> torch.Tensor:
    """Sum log-prob of the response tokens for each sequence in the batch.

    ``input_ids`` and ``labels`` follow the same shift-by-one convention
    as the SFT collator: position t in labels is the gold next-token for
    position t in input_ids; pad / prompt positions are IGNORE_INDEX.

    Returns a (B,) tensor of summed log-probs (or per-token mean if
    ``length_normalize`` is True). Differentiable wrt model parameters.
    """
    raw_model = _unwrap_model(model)
    if not hasattr(raw_model, "forward_hidden") or not hasattr(raw_model, "lm_head"):
        logits, _ = model(input_ids, return_logits=True)
        return _per_sequence_logp_from_logits(
            logits,
            labels,
            length_normalize=length_normalize,
        )

    hidden = raw_model.forward_hidden(input_ids)
    lm_head = raw_model.lm_head
    return _per_sequence_logp_from_hidden(
        hidden,
        labels,
        lm_head=lm_head,
        length_normalize=length_normalize,
        loss_chunk_rows=loss_chunk_rows,
    )


def _per_sequence_logp_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Reference full-logit path used by tests and fallback wrappers."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather the log-prob of each gold next-token. Replace -100 with 0 to
    # safely index, then mask their contributions out.
    safe_labels = shift_labels.clamp(min=0)
    per_token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != IGNORE_INDEX).float()
    per_token_logp = per_token_logp * mask

    if length_normalize:
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return per_token_logp.sum(dim=-1) / denom
    return per_token_logp.sum(dim=-1)


def _per_sequence_logp_from_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    lm_head: nn.Linear,
    length_normalize: bool = False,
    loss_chunk_rows: int = 2048,
) -> torch.Tensor:
    """Chunked response-token log-prob scoring without full sequence logits."""
    B = hidden.size(0)
    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.reshape(-1)
    seq_ids = torch.arange(B, device=hidden.device).unsqueeze(1)
    seq_ids = seq_ids.expand(B, shift_labels.size(1)).reshape(-1)

    total = torch.zeros(B, device=hidden.device, dtype=torch.float32)
    counts = torch.zeros(B, device=hidden.device, dtype=torch.float32)
    supervised = flat_labels != IGNORE_INDEX
    if not bool(supervised.any().item()):
        return total

    flat_hidden = flat_hidden[supervised]
    flat_labels = flat_labels[supervised]
    seq_ids = seq_ids[supervised]

    for start in range(0, flat_hidden.size(0), loss_chunk_rows):
        stop = start + loss_chunk_rows
        hidden_chunk = flat_hidden[start:stop]
        label_chunk = flat_labels[start:stop]
        seq_chunk = seq_ids[start:stop]
        logits_chunk = F.linear(hidden_chunk, lm_head.weight, lm_head.bias)
        log_probs = F.log_softmax(logits_chunk, dim=-1)
        vals = log_probs.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        total.index_add_(0, seq_chunk, vals.float())
        counts.index_add_(0, seq_chunk, torch.ones_like(vals, dtype=torch.float32))

    if length_normalize:
        return total / counts.clamp(min=1.0)
    return total


def _unwrap_model(model: nn.Module) -> nn.Module:
    raw = model
    if hasattr(raw, "module"):
        raw = raw.module
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


# ── Optimiser ────────────────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, dpo: DPOConfig) -> torch.optim.AdamW:
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    device_type = next(model.parameters()).device.type
    groups = [
        {"params": decay_params, "weight_decay": dpo.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {}
    if device_type == "cuda":
        kwargs["fused"] = True
    elif device_type == "mps":
        kwargs["foreach"] = False
        kwargs["fused"] = False
    return torch.optim.AdamW(
        groups,
        lr=dpo.learning_rate,
        betas=(dpo.beta1, dpo.beta2),
        **kwargs,
    )


# ── Mixed-precision context (mirrors SFT) ────────────────────────────────────

def _autocast_for(device: torch.device) -> tuple[Any, torch.amp.GradScaler | None]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:
            return torch.autocast("cuda", dtype=torch.bfloat16), None
        scaler = torch.amp.GradScaler("cuda")
        return torch.autocast("cuda", dtype=torch.float16), scaler
    return nullcontext(), None


# ── Model loading ────────────────────────────────────────────────────────────

def _load_sft_model(checkpoint_path: str, device: torch.device) -> tuple[ParrotLLM, dict]:
    log.info("Loading SFT checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} schema invalid (need 'model' + 'config')."
        )
    cfg = ckpt["config"]
    init_cfg = dict(cfg) if isinstance(cfg, dict) and "model" in cfg else {"model": dict(cfg)}
    model = ParrotLLM(init_cfg)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys: %s", missing[:5])
    if unexpected:
        log.warning("Unexpected keys: %s", unexpected[:5])
    model.to(device)
    log.info("Loaded model with %.2fM params", model.count_parameters() / 1e6)
    return model, dict(cfg) if isinstance(cfg, dict) else {}


# ── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    policy,
    reference,
    val_loader,
    device,
    autocast,
    *,
    beta,
    length_normalize,
    loss_chunk_rows: int = 2048,
):
    policy.eval()
    losses, accs = [], []
    for batch in val_loader:
        ci = batch["chosen_input_ids"].to(device, non_blocking=True)
        cl = batch["chosen_labels"].to(device, non_blocking=True)
        ri = batch["rejected_input_ids"].to(device, non_blocking=True)
        rl = batch["rejected_labels"].to(device, non_blocking=True)
        with autocast:
            pol_c = per_sequence_logp(
                policy, ci, cl,
                length_normalize=length_normalize,
                loss_chunk_rows=loss_chunk_rows,
            )
            pol_r = per_sequence_logp(
                policy, ri, rl,
                length_normalize=length_normalize,
                loss_chunk_rows=loss_chunk_rows,
            )
            ref_c = per_sequence_logp(
                reference, ci, cl,
                length_normalize=length_normalize,
                loss_chunk_rows=loss_chunk_rows,
            )
            ref_r = per_sequence_logp(
                reference, ri, rl,
                length_normalize=length_normalize,
                loss_chunk_rows=loss_chunk_rows,
            )
        loss, m = dpo_loss(
            policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
            ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=beta,
        )
        losses.append(loss.item())
        accs.append(m["accuracy"])
    policy.train()
    if not losses:
        return {"val_loss": float("nan"), "val_accuracy": float("nan")}
    return {"val_loss": sum(losses) / len(losses),
            "val_accuracy": sum(accs) / len(accs)}


# ── Run state ────────────────────────────────────────────────────────────────

@dataclass
class DPORunState:
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_checkpoint: str | None = None
    bad_evals: int = 0


def run_dpo(
    project_config: ProjectConfig,
    *,
    checkpoint: str,
    device: torch.device | None = None,
) -> None:
    """Run the DPO stage end-to-end."""
    dpo: DPOConfig = project_config.dpo  # type: ignore[attr-defined]
    if dpo is None:
        raise ValueError("Configuration section 'dpo' is missing. Add a `dpo:` block to YAML.")
    if not checkpoint:
        raise ValueError("DPO requires an SFT checkpoint via --checkpoint.")
    if device is None:
        device = get_device(dpo.device)
    log.info("DPO: device=%s, beta=%.3f, lr=%.2e", device, dpo.beta, dpo.learning_rate)

    run_dir = make_run_dir(dpo.runs_dir, tag="dpo")
    log.info("DPO run directory: %s", run_dir)
    metrics_path = Path(run_dir) / "metrics.jsonl"

    tokenizer = build_tokenizer()
    pad_id = tokenizer.pad_token_id
    log.info("Tokenizer: vocab=%d, pad_id=%d, eos_id=%d",
             len(tokenizer), pad_id, tokenizer.eos_token_id)

    # Decontamination
    decontam_texts: list[str] | None = None
    if dpo.decontam_benchmarks:
        decontam_texts = list(
            load_decontam_texts(
                dpo.decontam_benchmarks,
                cleanup_hf_cache=dpo.cleanup_hf_cache,
            )
        )
        log.info("Decontam: collected %d benchmark strings.", len(decontam_texts))

    bundle = build_dpo_datasets(
        hf_dataset_name=dpo.hf_dataset_name,
        hf_split=dpo.hf_split,
        tokenizer=tokenizer,
        max_length=dpo.max_length,
        val_fraction=dpo.val_fraction,
        seed=42,
        decontam_texts=decontam_texts,
        max_examples=dpo.max_examples,
        preference_jsonl_path=dpo.preference_jsonl_path,
        preference_oversample=dpo.preference_oversample,
        hf_cache_dir=dpo.hf_cache_dir,
        cleanup_hf_cache=dpo.cleanup_hf_cache,
    )
    log.info("Dataset ready: train=%d, val=%d", len(bundle.train), len(bundle.val))

    collator = DPOCollator(
        pad_token_id=pad_id,
        max_length=dpo.max_length,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    train_loader = DataLoader(
        bundle.train, batch_size=dpo.batch_size, shuffle=True,
        num_workers=dpo.num_workers,
        pin_memory=dpo.pin_memory and device.type == "cuda",
        collate_fn=collator, drop_last=True,
    )
    val_loader = DataLoader(
        bundle.val, batch_size=max(1, dpo.batch_size), shuffle=False,
        num_workers=0, pin_memory=dpo.pin_memory and device.type == "cuda",
        collate_fn=collator,
    )

    # Load SFT checkpoint TWICE: policy (trainable) + reference (frozen).
    policy, embedded_config = _load_sft_model(checkpoint, device)
    reference, _ = _load_sft_model(checkpoint, device)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    policy.train()

    # 5090 perf knobs (mirror SFT)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if dpo.torch_compile and device.type == "cuda":
        log.info("Compiling policy + reference (torch.compile, mode='default')...")
        policy = torch.compile(policy, fullgraph=False)
        reference = torch.compile(reference, fullgraph=False)

    optimizer = _build_optimizer(policy, dpo)
    steps_per_epoch = max(1, len(train_loader) // dpo.gradient_accumulation_steps)
    total_optim_steps = steps_per_epoch * dpo.epochs
    log.info(
        "Training plan: %d epochs × %d optim steps/epoch = %d total "
        "(batch_size=%d × grad_accum=%d → effective batch=%d)",
        dpo.epochs, steps_per_epoch, total_optim_steps,
        dpo.batch_size, dpo.gradient_accumulation_steps,
        dpo.batch_size * dpo.gradient_accumulation_steps,
    )

    autocast, scaler = _autocast_for(device)
    state = DPORunState()

    def _write_metric(payload: dict) -> None:
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(payload) + "\n")

    # WT-103 CF baseline
    wt103_tokens = None
    wt103_baseline_ppl = None
    if dpo.wt103_eval_every_n_evals > 0:
        log.info("Loading Wikitext-103 for CF tripwire...")
        wt103_tokens = _load_wt103_tokens(
            tokenizer, max_tokens=(dpo.wt103_max_sequences + 2) * dpo.max_length,
            cleanup_hf_cache=dpo.cleanup_hf_cache,
        )
        wt103_baseline_ppl = compute_perplexity(
            policy, wt103_tokens, dpo.max_length, device,
            batch_size=max(1, dpo.batch_size),
            max_sequences=dpo.wt103_max_sequences,
        )
        log.info("WT-103 baseline PPL on SFT model: %.3f", wt103_baseline_ppl)

    # Preflight: sanity check on one batch.
    first = next(iter(train_loader))
    n_chosen = count_supervised_tokens(first["chosen_labels"])
    n_rejected = count_supervised_tokens(first["rejected_labels"])
    log.info("Preflight: chosen=%d / rejected=%d supervised tokens.",
             n_chosen, n_rejected)
    if n_chosen == 0 or n_rejected == 0:
        raise RuntimeError(
            "Preflight failed: one half of the preference pair has 0 supervised "
            "tokens. The DPO collator mask is wrong; check tokenise_dpo_example."
        )

    # ── Training loop ──────────────────────────────────────────────────────
    start_time = time.time()
    accum_counter = 0
    accum_loss_sum = 0.0
    accum_acc_sum = 0.0
    nonfinite_streak = 0
    oom_streak = 0
    should_stop_early = False

    for epoch in range(dpo.epochs):
        state.epoch = epoch
        log.info("── Epoch %d/%d ──", epoch + 1, dpo.epochs)
        for batch in train_loader:
            ci = batch["chosen_input_ids"].to(device, non_blocking=True)
            cl = batch["chosen_labels"].to(device, non_blocking=True)
            ri = batch["rejected_input_ids"].to(device, non_blocking=True)
            rl = batch["rejected_labels"].to(device, non_blocking=True)

            try:
                with autocast:
                    pol_c = per_sequence_logp(
                        policy, ci, cl,
                        length_normalize=dpo.length_normalize_logp,
                        loss_chunk_rows=dpo.loss_chunk_rows,
                    )
                    pol_r = per_sequence_logp(
                        policy, ri, rl,
                        length_normalize=dpo.length_normalize_logp,
                        loss_chunk_rows=dpo.loss_chunk_rows,
                    )
                    with torch.no_grad():
                        ref_c = per_sequence_logp(
                            reference, ci, cl,
                            length_normalize=dpo.length_normalize_logp,
                            loss_chunk_rows=dpo.loss_chunk_rows,
                        )
                        ref_r = per_sequence_logp(
                            reference, ri, rl,
                            length_normalize=dpo.length_normalize_logp,
                            loss_chunk_rows=dpo.loss_chunk_rows,
                        )
                    loss, m = dpo_loss(
                        policy_chosen_logp=pol_c, policy_rejected_logp=pol_r,
                        ref_chosen_logp=ref_c, ref_rejected_logp=ref_r, beta=dpo.beta,
                    )
            except RuntimeError as exc:
                if not _is_recoverable_mps_oom(exc, device):
                    raise
                log.warning(
                    "MPS OOM during DPO forward at step %d. Clearing cache, "
                    "dropping the partial accumulation window, and continuing: %s",
                    state.step, exc,
                )
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                accum_loss_sum = 0.0
                accum_acc_sum = 0.0
                oom_streak += 1
                _empty_device_cache(device)
                if oom_streak >= 5:
                    raise RuntimeError(
                        "DPO saw 5 consecutive recoverable MPS OOM windows. "
                        "Lower dpo.max_length or close memory-heavy apps."
                    ) from exc
                continue

            if _is_nonfinite(loss):
                log.warning(
                    "Non-finite DPO loss at step %d. Skipping batch and resetting accumulator.",
                    state.step,
                )
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                accum_loss_sum = 0.0
                accum_acc_sum = 0.0
                nonfinite_streak += 1
                _empty_device_cache(device)
                if nonfinite_streak >= 25:
                    raise RuntimeError(
                        "DPO saw 25 consecutive non-finite batches. Stopping "
                        "instead of looping forever."
                    )
                continue

            loss_value = loss.item()
            nonfinite_streak = 0
            loss_to_back = loss / dpo.gradient_accumulation_steps
            try:
                if scaler is not None:
                    scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()
            except RuntimeError as exc:
                if not _is_recoverable_mps_oom(exc, device):
                    raise
                log.warning(
                    "MPS OOM during DPO backward at step %d. Clearing cache, "
                    "dropping the partial accumulation window, and continuing: %s",
                    state.step, exc,
                )
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                accum_loss_sum = 0.0
                accum_acc_sum = 0.0
                oom_streak += 1
                del loss_to_back
                del loss
                _empty_device_cache(device)
                if oom_streak >= 5:
                    raise RuntimeError(
                        "DPO saw 5 consecutive recoverable MPS OOM windows. "
                        "Lower dpo.max_length or close memory-heavy apps."
                    ) from exc
                continue
            del loss_to_back
            if device.type == "mps":
                _empty_device_cache(device)
            oom_streak = 0

            accum_loss_sum += loss_value
            accum_acc_sum += m["accuracy"]
            accum_counter += 1

            if accum_counter < dpo.gradient_accumulation_steps:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            if dpo.grad_clip > 0:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        max_norm=dpo.grad_clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    log.warning(
                        "Non-finite DPO gradient norm at step %d. Skipping "
                        "optimizer step and resetting accumulator: %s",
                        state.step, exc,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    accum_loss_sum = 0.0
                    accum_acc_sum = 0.0
                    nonfinite_streak += 1
                    _empty_device_cache(device)
                    if nonfinite_streak >= 25:
                        raise RuntimeError(
                            "DPO saw 25 consecutive non-finite gradient windows."
                        ) from exc
                    continue

            if dpo.lr_schedule == "cosine":
                lr = _cosine_lr(state.step, dpo.warmup_steps, total_optim_steps,
                                dpo.learning_rate, dpo.min_lr)
            else:
                lr = _wsd_lr(state.step, dpo.warmup_steps, total_optim_steps,
                             dpo.learning_rate, dpo.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not _model_parameters_are_finite(policy):
                optimizer.zero_grad(set_to_none=True)
                _empty_device_cache(device)
                raise RuntimeError(
                    "DPO optimizer step produced NaN/Inf model parameters. "
                    "Aborting before the checkpoint is corrupted."
                )
            optimizer.zero_grad(set_to_none=True)
            _empty_device_cache(device)
            state.step += 1

            avg_loss = accum_loss_sum / accum_counter
            avg_acc = accum_acc_sum / accum_counter

            if state.step % dpo.log_every == 0:
                elapsed = time.time() - start_time
                log.info(
                    "step %d/%d | dpo_loss %.4f | acc %.3f | margin %.3f | lr %.2e | %.0f s",
                    state.step, total_optim_steps, avg_loss, avg_acc,
                    m["reward_margin"], lr, elapsed,
                )
                _write_metric({
                    "stage": "dpo",
                    "step": state.step,
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "reward_margin": m["reward_margin"],
                    "rewards_chosen": m["rewards_chosen"],
                    "rewards_rejected": m["rewards_rejected"],
                    "lr": lr,
                })

            if state.step % dpo.eval_every == 0:
                val = _evaluate(
                    policy, reference, val_loader, device, autocast,
                    beta=dpo.beta,
                    length_normalize=dpo.length_normalize_logp,
                    loss_chunk_rows=dpo.loss_chunk_rows,
                )
                _empty_device_cache(device)
                log.info("  [val] dpo_loss %.4f | acc %.3f",
                         val["val_loss"], val["val_accuracy"])
                _write_metric({"stage": "dpo", "step": state.step, "epoch": epoch, **val})
                if val["val_loss"] < state.best_val_loss:
                    state.best_val_loss = val["val_loss"]
                    state.bad_evals = 0
                    state.best_checkpoint = _save_checkpoint(
                        model=policy, run_dir=run_dir, tag="best",
                        step=state.step, epoch=epoch,
                        val_loss=val["val_loss"],
                        embedded_config=embedded_config, dpo_config=dpo,
                        stats=bundle.stats,
                    )
                else:
                    state.bad_evals += 1
                    if _should_stop_early(state.bad_evals, dpo.early_stopping_patience):
                        log.info(
                            "Early stop: %d non-improving evals (patience=%d). Best val=%.4f at %s.",
                            state.bad_evals, dpo.early_stopping_patience,
                            state.best_val_loss, state.best_checkpoint,
                        )
                        should_stop_early = True

                # WT-103 tripwire
                if (wt103_tokens is not None and wt103_baseline_ppl is not None
                        and dpo.wt103_eval_every_n_evals > 0):
                    eval_idx = state.step // dpo.eval_every
                    if eval_idx % dpo.wt103_eval_every_n_evals == 0:
                        cur_ppl = compute_perplexity(
                            policy, wt103_tokens, dpo.max_length, device,
                            batch_size=max(1, dpo.batch_size),
                            max_sequences=dpo.wt103_max_sequences,
                        )
                        delta_pct = 100.0 * (cur_ppl - wt103_baseline_ppl) / wt103_baseline_ppl
                        log.info(
                            "  [wt103] PPL %.3f (Δ %+.2f%% vs SFT-baseline %.3f)",
                            cur_ppl, delta_pct, wt103_baseline_ppl,
                        )
                        _write_metric({
                            "stage": "dpo", "step": state.step,
                            "wt103_ppl": cur_ppl,
                            "wt103_baseline_ppl": wt103_baseline_ppl,
                            "wt103_delta_pct": delta_pct,
                        })
                        if _wt103_should_stop(
                            baseline_ppl=wt103_baseline_ppl, current_ppl=cur_ppl,
                            threshold_pct=dpo.wt103_hard_stop_pct,
                        ):
                            log.warning(
                                "CF tripwire: WT-103 PPL up %.2f%% ≥ %.2f%% threshold. "
                                "Halting DPO. Best val=%.4f at %s.",
                                delta_pct, dpo.wt103_hard_stop_pct,
                                state.best_val_loss, state.best_checkpoint,
                            )
                            should_stop_early = True

            accum_counter = 0
            accum_loss_sum = 0.0
            accum_acc_sum = 0.0

            if state.step >= total_optim_steps or should_stop_early:
                break

        if accum_counter > 0:
            log.info("Discarding %d partial-accumulation batches at end of epoch %d.",
                     accum_counter, epoch + 1)
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            accum_loss_sum = 0.0
            accum_acc_sum = 0.0

        if state.step >= total_optim_steps or should_stop_early:
            break

    final_val = _evaluate(
        policy, reference, val_loader, device, autocast,
        beta=dpo.beta,
        length_normalize=dpo.length_normalize_logp,
        loss_chunk_rows=dpo.loss_chunk_rows,
    )
    log.info("DPO complete. Final val loss %.4f acc %.3f (best %.4f). Total steps %d.",
             final_val["val_loss"], final_val["val_accuracy"],
             state.best_val_loss, state.step)
    _save_checkpoint(
        model=policy, run_dir=run_dir, tag="final",
        step=state.step, epoch=state.epoch,
        val_loss=final_val["val_loss"],
        embedded_config=embedded_config, dpo_config=dpo,
        stats=bundle.stats,
    )


def _save_checkpoint(*, model, run_dir, tag, step, epoch, val_loss,
                     embedded_config, dpo_config: DPOConfig, stats) -> str:
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vl = "na" if val_loss is None else f"{val_loss:.4f}".replace(".", "p")
    fname = f"{tag}_step_{step:07d}_epoch_{epoch:02d}_valloss_{vl}.pt"
    path = ckpt_dir / fname

    raw_model = model
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    payload = {
        "model": raw_model.state_dict(),
        "config": embedded_config,
        "step": step,
        "epoch": epoch,
        "training_stage": "dpo",
        "dpo_metadata": {
            "tag": tag,
            "val_loss": val_loss,
            "dataset": dpo_config.hf_dataset_name,
            "split": dpo_config.hf_split,
            "max_length": dpo_config.max_length,
            "epochs": dpo_config.epochs,
            "learning_rate": dpo_config.learning_rate,
            "beta": dpo_config.beta,
            "batch_size": dpo_config.batch_size,
            "grad_accum": dpo_config.gradient_accumulation_steps,
            "length_normalize_logp": dpo_config.length_normalize_logp,
            "stats": stats,
        },
    }
    torch.save(payload, path)
    log.info("Saved DPO checkpoint: %s", path)
    return str(path)
