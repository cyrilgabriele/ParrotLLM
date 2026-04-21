"""SFT training loop for ParrotLLM.

Implements VL07's "Golden Loop" (VL04 slide 43 originally, re-used for SFT):

    1. Forward pass in mixed precision
    2. Cross-entropy loss on RESPONSE tokens only (masked, VL07 slide 15)
    3. Divide by K (gradient accumulation)
    4. Backward pass
    5. At accumulation step K:
         - Clip gradient norm ≤ 1.0 (VL04 "Gradient Clipping")
         - AdamW step with LR schedule (VL04 slide 33 cosine/warmup)
         - Zero gradients
    6. Loop back
    7. Every N steps: evaluate on held-out val split
    8. Save checkpoint with `training_stage="sft"`

Design decisions (documented in docs/post_training/SFT.md):

- We START FROM A PRETRAINING CHECKPOINT, always. SFT on random weights
  makes no sense; VL07 slide 12 is explicit that SFT teaches behaviour to
  a model that already knows language. The trainer refuses to run without
  `--checkpoint`.

- We do NOT reuse the pretraining trainer (`src/training/trainer.py`,
  1664 lines) verbatim because it is built around `.bin` token arrays
  with shifted (inputs, targets) tuples. SFT needs padded batches with
  HF-style `labels` (-100 mask). The simplest correct design is to
  implement a lean loop here and extend the model's forward pass to
  accept `labels` (done in `src/model/transformer.py`).

- Mixed precision: BF16 on Ampere+ (per VL04 "Double the Speed"); FP16 +
  GradScaler on older GPUs; plain FP32 on CPU/MPS. This mirrors the
  pretraining trainer exactly so a checkpoint trained in one stage is
  trivially loadable in the other.

- Weight decay: 0.1 on 2D params (weight matrices), 0.0 on 1D params
  (biases, norms, embeddings). Same split as pretraining — VL04 covers
  the rationale ("AdamW recipe card").

- Logging: JSONL per step + a minimal metrics CSV. Re-using
  `src.logging_utils.make_run_dir` keeps SFT runs visible in the same
  dashboard as pretraining runs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import ProjectConfig
from configs.post_training.sftConfig import SFTConfig
from src.logging_utils import init_logging, make_run_dir
from src.model import ParrotLLM
from src.post_training.sft.collator import IGNORE_INDEX, SFTCollator, count_supervised_tokens
from src.post_training.sft.data import build_sft_datasets
from src.post_training.sft.template import DEFAULT_ALPACA_TEMPLATE
from src.utils import build_tokenizer, get_device


log = logging.getLogger("parrotllm.sft.trainer")


# ── LR schedule (inlined — small enough to duplicate cleanly) ────────────────

def _cosine_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    """Linear warmup → cosine decay to ``floor``.

    Matches the pretraining scheduler family (VL04 slide 33). We inline it
    here instead of importing ``ParrotLRScheduler`` from the pretraining
    module to avoid coupling — if the pretraining LR code changes we do
    not want SFT silently drifting with it.
    """
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cos


def _wsd_lr(step: int, warmup: int, total: int, peak: float, floor: float,
            decay_ratio: float = 0.1) -> float:
    """Warmup → stable → linear-decay, per our pretraining WSD schedule."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    decay_steps = max(1, int(total * decay_ratio))
    decay_start = total - decay_steps
    if step < decay_start:
        return peak
    if step >= total:
        return floor
    progress = (step - decay_start) / decay_steps
    return peak - (peak - floor) * progress


# ── Optimiser ────────────────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, sft: SFTConfig) -> torch.optim.AdamW:
    """AdamW with weight-decay only on 2D parameters (VL04 "AdamW recipe")."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": sft.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=sft.learning_rate,
        betas=(sft.beta1, sft.beta2),
        fused=torch.cuda.is_available(),
    )


# ── Mixed-precision context (mirrors pretraining trainer) ────────────────────

def _autocast_for(device: torch.device) -> tuple[Any, torch.amp.GradScaler | None]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:  # Ampere+, includes 5090 (RTX 40/50 series: major >= 8)
            return torch.autocast("cuda", dtype=torch.bfloat16), None
        scaler = torch.amp.GradScaler("cuda")
        return torch.autocast("cuda", dtype=torch.float16), scaler
    return nullcontext(), None


# ── Model loading from pretraining checkpoint ────────────────────────────────

def _load_base_model(checkpoint_path: str, device: torch.device) -> tuple[ParrotLLM, dict]:
    """Load a pretraining checkpoint, return (model, embedded_config_dict).

    Expects the ParrotLLM checkpoint schema: ``{"config": {...}, "model":
    state_dict, ...}``. Raises loudly if the schema is wrong so SFT does
    not silently run on randomly-initialised weights (the single worst
    possible failure mode here).
    """
    log.info("Loading base pretraining checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not match the ParrotLLM "
            "schema (expected keys 'model' and 'config'). SFT refuses "
            "to initialise from an unrecognised checkpoint."
        )
    cfg = ckpt["config"]
    # ParrotLLM.__init__ expects a project-config-shaped dict with a "model"
    # subkey (see src/model/transformer.py line 160). A full saved config has
    # that shape already; if we only got the inner model sub-dict (as some
    # unit tests do), wrap it.
    if isinstance(cfg, dict) and "model" in cfg:
        init_cfg = dict(cfg)
    else:
        init_cfg = {"model": dict(cfg)}
    model = ParrotLLM(init_cfg)
    state_dict = ckpt["model"]
    # Strip possible `_orig_mod.` prefix added by torch.compile.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys when loading base checkpoint: %s", missing[:5])
    if unexpected:
        log.warning("Unexpected keys in base checkpoint: %s", unexpected[:5])
    model.to(device)
    param_count = model.count_parameters()
    log.info("Base model loaded. Parameters: %d (%.2fM)", param_count, param_count / 1e6)
    return model, dict(cfg) if isinstance(cfg, dict) else {}


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    autocast,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation: mean loss per supervised token."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast:
            loss = model(input_ids, labels=labels)
        n_supervised = count_supervised_tokens(labels)
        if n_supervised == 0:
            continue
        total_loss += loss.item() * n_supervised
        total_tokens += n_supervised
    model.train()
    if total_tokens == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan"), "val_tokens": 0.0}
    mean_loss = total_loss / total_tokens
    return {
        "val_loss": mean_loss,
        "val_ppl": math.exp(min(mean_loss, 20.0)),
        "val_tokens": float(total_tokens),
    }


# ── Main entry ───────────────────────────────────────────────────────────────

@dataclass
class SFTRunState:
    """Mutable training progress for JSONL logging."""

    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    best_checkpoint: str | None = None


def run_sft(
    project_config: ProjectConfig,
    *,
    checkpoint: str,
    device: torch.device | None = None,
) -> None:
    """Run the SFT stage end-to-end.

    Args:
        project_config: the loaded project config. Must have a ``sft``
            section (see ``configs/default.yaml``).
        checkpoint: path to the pretraining checkpoint to initialise from.
            Required — SFT on random weights is meaningless (VL07 slide 12).
        device: torch device; ``None`` → auto-detect.
    """
    sft: SFTConfig = project_config.sft  # type: ignore[attr-defined]
    if sft is None:
        raise ValueError(
            "Configuration section 'sft' is missing. Add an `sft:` block to "
            "configs/default.yaml."
        )
    if not checkpoint:
        raise ValueError(
            "SFT requires a base pretraining checkpoint via --checkpoint. "
            "Starting from random weights breaks the whole point of SFT."
        )
    if device is None:
        device = get_device(sft.device)
    log.info("SFT: device=%s", device)

    # ── Run dir ────────────────────────────────────────────────────────────
    run_dir = make_run_dir(sft.runs_dir, tag="sft")
    log.info("SFT run directory: %s", run_dir)
    metrics_path = Path(run_dir) / "metrics.jsonl"

    # ── Tokeniser ──────────────────────────────────────────────────────────
    # Same tokeniser as pretraining: GPT-2 + <|pad|>. See src/utils.py.
    # VL07 slide 32's "critical rule" — train-time and inference-time
    # tokenisation must agree — reduces to using exactly this function.
    tokenizer = build_tokenizer()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    log.info("Tokenizer: vocab=%d, pad_id=%d, eos_id=%d",
             len(tokenizer), pad_id, eos_id)

    # ── Dataset ────────────────────────────────────────────────────────────
    bundle = build_sft_datasets(
        hf_dataset_name=sft.hf_dataset_name,
        hf_split=sft.hf_split,
        tokenizer=tokenizer,
        template=DEFAULT_ALPACA_TEMPLATE,
        max_length=sft.max_length,
        val_fraction=sft.val_fraction,
        seed=42,
        decontam_texts=None,  # wire up from sft.decontam_benchmarks later
        max_examples=sft.max_examples,
    )
    log.info("Dataset ready: train=%d, val=%d", len(bundle.train), len(bundle.val))

    collator = SFTCollator(
        pad_token_id=pad_id,
        max_length=sft.max_length,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    train_loader = DataLoader(
        bundle.train,
        batch_size=sft.batch_size,
        shuffle=True,
        num_workers=sft.num_workers,
        pin_memory=sft.pin_memory and device.type == "cuda",
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=max(1, sft.batch_size),
        shuffle=False,
        num_workers=0,  # tiny corpus; avoid worker overhead
        pin_memory=sft.pin_memory and device.type == "cuda",
        collate_fn=collator,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model, embedded_config = _load_base_model(checkpoint, device)
    model.train()

    # ── Optimiser + schedule ───────────────────────────────────────────────
    optimizer = _build_optimizer(model, sft)
    steps_per_epoch = max(1, len(train_loader) // sft.gradient_accumulation_steps)
    total_optim_steps = steps_per_epoch * sft.epochs
    log.info(
        "Training plan: %d epochs × %d optim steps/epoch = %d total optim steps "
        "(batch_size=%d × grad_accum=%d → effective batch=%d)",
        sft.epochs, steps_per_epoch, total_optim_steps,
        sft.batch_size, sft.gradient_accumulation_steps,
        sft.batch_size * sft.gradient_accumulation_steps,
    )

    autocast, scaler = _autocast_for(device)

    state = SFTRunState()

    def _write_metric(payload: dict) -> None:
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(payload) + "\n")

    # Preflight: one forward on one batch, verify mask is non-trivial.
    # This is VL07 slide 6's "Tale of Two Students" lesson made operational
    # — catch the silent failure (mask wrong → loss computed over 0 tokens
    # → gradients zero → "training" for hours with no actual learning).
    first = next(iter(train_loader))
    n_sup = count_supervised_tokens(first["labels"])
    total_pos = first["input_ids"].numel()
    log.info(
        "Preflight batch: %d supervised tokens / %d total (%.1f%% supervised).",
        n_sup, total_pos, 100 * n_sup / max(1, total_pos),
    )
    if n_sup == 0:
        raise RuntimeError(
            "Preflight check failed: 0 supervised tokens in the first batch. "
            "The label mask is wrong. See VL07 slide 15; check "
            "SFTCollator and tokenise_example."
        )

    # ── Training loop ──────────────────────────────────────────────────────
    start_time = time.time()
    accum_counter = 0
    accum_loss_sum = 0.0
    accum_tokens_sum = 0

    for epoch in range(sft.epochs):
        state.epoch = epoch
        log.info("── Epoch %d/%d ──", epoch + 1, sft.epochs)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast:
                loss = model(input_ids, labels=labels)
            loss_value = loss.item()

            # Gradient accumulation: divide the per-step loss by K so the
            # accumulated gradient has the correct magnitude after K backward
            # passes. This is VL04 slide 39 rule #3 ("Important: divide loss
            # by K, otherwise gradients are K times too large").
            loss_to_back = loss / sft.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            n_tokens = count_supervised_tokens(labels)
            accum_loss_sum += loss_value * n_tokens
            accum_tokens_sum += n_tokens
            accum_counter += 1
            state.tokens_seen += n_tokens

            if accum_counter < sft.gradient_accumulation_steps:
                continue

            # One optimiser step.
            if scaler is not None:
                scaler.unscale_(optimizer)
            if sft.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=sft.grad_clip,
                )
            if sft.lr_schedule == "cosine":
                lr = _cosine_lr(state.step, sft.warmup_steps, total_optim_steps,
                                sft.learning_rate, sft.min_lr)
            else:
                lr = _wsd_lr(state.step, sft.warmup_steps, total_optim_steps,
                             sft.learning_rate, sft.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            state.step += 1
            accum_loss_per_token = (
                accum_loss_sum / accum_tokens_sum if accum_tokens_sum > 0 else float("nan")
            )

            if state.step % sft.log_every == 0:
                elapsed = time.time() - start_time
                tok_per_s = state.tokens_seen / max(elapsed, 1e-6)
                log.info(
                    "step %d/%d | loss %.4f | ppl %.2f | lr %.2e | tok/s %.0f",
                    state.step, total_optim_steps,
                    accum_loss_per_token,
                    math.exp(min(accum_loss_per_token, 20.0)),
                    lr, tok_per_s,
                )
                _write_metric({
                    "stage": "sft",
                    "step": state.step,
                    "epoch": epoch,
                    "train_loss": accum_loss_per_token,
                    "lr": lr,
                    "tokens_seen": state.tokens_seen,
                    "tok_per_s": tok_per_s,
                })

            if state.step % sft.eval_every == 0:
                val_metrics = _evaluate(model, val_loader, device, autocast)
                log.info(
                    "  [val] loss %.4f | ppl %.2f | tokens %d",
                    val_metrics["val_loss"], val_metrics["val_ppl"],
                    int(val_metrics["val_tokens"]),
                )
                _write_metric({
                    "stage": "sft",
                    "step": state.step,
                    "epoch": epoch,
                    **val_metrics,
                })
                if val_metrics["val_loss"] < state.best_val_loss:
                    state.best_val_loss = val_metrics["val_loss"]
                    state.best_checkpoint = _save_checkpoint(
                        model=model,
                        run_dir=run_dir,
                        tag="best",
                        step=state.step,
                        epoch=epoch,
                        val_loss=val_metrics["val_loss"],
                        embedded_config=embedded_config,
                        sft_config=sft,
                        stats=bundle.stats,
                    )

            if state.step % sft.save_every == 0:
                _save_checkpoint(
                    model=model,
                    run_dir=run_dir,
                    tag="last",
                    step=state.step,
                    epoch=epoch,
                    val_loss=None,
                    embedded_config=embedded_config,
                    sft_config=sft,
                    stats=bundle.stats,
                )

            accum_counter = 0
            accum_loss_sum = 0.0
            accum_tokens_sum = 0

            if state.step >= total_optim_steps:
                break
        if state.step >= total_optim_steps:
            break

    # Final checkpoint + summary.
    final_val = _evaluate(model, val_loader, device, autocast)
    log.info(
        "SFT complete. Final val loss %.4f (best %.4f). Total steps %d, tokens %d.",
        final_val["val_loss"], state.best_val_loss, state.step, state.tokens_seen,
    )
    _save_checkpoint(
        model=model,
        run_dir=run_dir,
        tag="final",
        step=state.step,
        epoch=state.epoch,
        val_loss=final_val["val_loss"],
        embedded_config=embedded_config,
        sft_config=sft,
        stats=bundle.stats,
    )


def _save_checkpoint(
    *,
    model: nn.Module,
    run_dir: str,
    tag: str,
    step: int,
    epoch: int,
    val_loss: float | None,
    embedded_config: dict,
    sft_config: SFTConfig,
    stats: dict,
) -> str:
    """Save an SFT checkpoint with a training_stage marker.

    The schema matches pretraining's ``save_checkpoint`` enough that Pair B's
    DPO loader can read it without surgery: `{config, model, step, ...}`.
    The added field is ``training_stage = "sft"`` plus a ``sft_metadata``
    block, per the hand-off contract in TEAM_SPLIT.md §"Inter-pair contracts".
    """
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
        "training_stage": "sft",
        "sft_metadata": {
            "tag": tag,
            "val_loss": val_loss,
            "dataset": sft_config.hf_dataset_name,
            "split": sft_config.hf_split,
            "max_length": sft_config.max_length,
            "epochs": sft_config.epochs,
            "learning_rate": sft_config.learning_rate,
            "batch_size": sft_config.batch_size,
            "grad_accum": sft_config.gradient_accumulation_steps,
            "weight_decay": sft_config.weight_decay,
            "template": "alpaca",
            "stats": stats,
        },
    }
    torch.save(payload, path)
    log.info("Saved SFT checkpoint: %s", path)
    return str(path)
