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

import gc
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import ProjectConfig
from configs.post_training.sftConfig import SFTConfig
from src.logging_utils import init_logging, make_run_dir
from src.model import ParrotLLM
from src.eval.perplexity import compute_perplexity
from src.post_training.sft.collator import IGNORE_INDEX, SFTCollator
from src.post_training.sft.data import build_sft_datasets, load_decontam_texts
from src.post_training.sft.template import DEFAULT_ALPACA_TEMPLATE
from src.post_training.hf_cache import cleanup_hf_dataset_cache
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


# ── Numerical guards ─────────────────────────────────────────────────────────

def _wt103_should_stop(
    *,
    baseline_ppl: float,
    current_ppl: float,
    threshold_pct: float,
) -> bool:
    """Return True when WT-103 perplexity has risen ≥ ``threshold_pct``
    above the base-checkpoint baseline.

    This implements the catastrophic-forgetting hard-stop described in
    SFT.md §6: if the model loses world knowledge faster than it acquires
    instruction-following, kill the run. ``threshold_pct == 0`` disables
    the tripwire (useful for ablation runs where you want to *measure*
    CF rather than be protected from it).
    """
    if threshold_pct <= 0:
        return False
    delta_pct = 100.0 * (current_ppl - baseline_ppl) / baseline_ppl
    return delta_pct >= threshold_pct


def _should_stop_early(bad_evals: int, patience: int) -> bool:
    """Return True when ``bad_evals`` consecutive non-improving evals
    have accumulated and early stopping is enabled.

    ``patience == 0`` disables early stopping entirely (treated as
    "never stop"). Reference: SFT.md §3.4 lists early stopping as
    catastrophic-forgetting mitigation #2.
    """
    return patience > 0 and bad_evals >= patience


def _is_nonfinite(loss: torch.Tensor) -> bool:
    """True if the loss is NaN or ±inf.

    A single non-finite loss propagated into AdamW corrupts the first- and
    second-moment buffers permanently — the rest of the run silently
    produces useless updates. The training loop checks this BEFORE
    `.backward()` and discards the in-flight accumulation window if hit.
    Documented failure mode of all-masked batches (test_sft_forward.py
    `test_all_positions_masked_produces_nan_loss`) and FP16 overflow.
    """
    return not bool(torch.isfinite(loss).item())


def _count_shifted_supervised_tokens(labels: torch.Tensor) -> int:
    """Count labels that are actually scored after the next-token shift."""
    if labels.ndim != 2 or labels.size(1) <= 1:
        return 0
    return int((labels[:, 1:] != IGNORE_INDEX).sum().item())


def _empty_device_cache(device: torch.device) -> None:
    """Release cached allocator blocks after skipped/eval batches."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()
    gc.collect()


def _is_recoverable_mps_oom(exc: RuntimeError, device: torch.device) -> bool:
    message = str(exc).lower()
    return device.type == "mps" and "out of memory" in message


def _model_parameters_are_finite(model: nn.Module) -> bool:
    """Check whether an optimizer step introduced NaN/Inf parameters."""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and not bool(torch.isfinite(param).all().item()):
                return False
    return True


# ── Optimiser ────────────────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, sft: SFTConfig) -> torch.optim.AdamW:
    """AdamW with weight-decay only on 2D parameters (VL04 "AdamW recipe")."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    device_type = next(model.parameters()).device.type
    groups = [
        {"params": decay_params, "weight_decay": sft.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {}
    if device_type == "cuda":
        kwargs["fused"] = True
    elif device_type == "mps":
        # The foreach/fused fast paths are not needed on Apple Silicon and can
        # be less stable across PyTorch nightlies. Use the conservative kernel.
        kwargs["foreach"] = False
        kwargs["fused"] = False
    return torch.optim.AdamW(
        groups,
        lr=sft.learning_rate,
        betas=(sft.beta1, sft.beta2),
        **kwargs,
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


# ── Pretraining-mix window sampler (CF mitigation #1, VL07 slide 25) ────────

def _draw_pretraining_window(
    memmap: np.ndarray,
    context_length: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows from a uint16 token array.

    Returns ``(inputs, targets)`` shaped ``(B, T)`` where ``targets`` is
    the next-token shift of ``inputs``. This matches the legacy pretraining
    forward path (``model(idx, targets=...)``) so the same model serves
    both the SFT loss (``labels=...``) and the CF-mitigation loss without
    branching.

    SFT.md §3.4 mitigation #3 / VL07 slide 25 #1: a small fraction of
    SFT batches are replaced with pretraining batches to keep the base
    model's distribution within reach.
    """
    n_tokens = len(memmap)
    max_start = n_tokens - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"Pretraining .bin too small for context_length={context_length}: "
            f"have {n_tokens} tokens, need ≥ {context_length + 1}."
        )
    starts = rng.integers(low=0, high=max_start + 1, size=batch_size)
    inputs = np.empty((batch_size, context_length), dtype=np.int64)
    targets = np.empty((batch_size, context_length), dtype=np.int64)
    for i, s in enumerate(starts):
        s = int(s)
        chunk = memmap[s : s + context_length + 1].astype(np.int64)
        inputs[i] = chunk[:-1]
        targets[i] = chunk[1:]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


# ── Wikitext-103 tokeniser cache for the CF tripwire ────────────────────────

def _load_wt103_tokens(
    tokenizer,
    max_tokens: int,
    *,
    cleanup_hf_cache: bool = False,
) -> torch.Tensor:
    """Load the Wikitext-103 test split, tokenise, slice to ``max_tokens``.

    Cached once at the start of the run; the same tensor is re-used for
    the baseline PPL on the base model and for every periodic re-eval.
    Slicing keeps the per-eval cost to seconds — full WT-103 test is
    ~245 k tokens which is overkill for a regression check.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer.encode(text)
    ds = None
    if cleanup_hf_cache:
        cleanup_hf_dataset_cache()
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long)


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
    loss_chunk_rows: int = 2048,
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
        n_supervised = _count_shifted_supervised_tokens(labels)
        if n_supervised == 0:
            continue
        with autocast:
            loss = model(
                input_ids,
                labels=labels,
                loss_chunk_rows=loss_chunk_rows,
            )
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
    bad_evals: int = 0  # consecutive non-improving evals (early-stop counter)


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

    # ── Decontamination corpus (SFT.md §3.3 "Mandatory") ──────────────────
    # Hash benchmark test texts so any verbatim leak in the SFT corpus is
    # dropped before training. Skipping this invalidates leaderboard scores.
    decontam_texts: list[str] | None = None
    if sft.decontam_benchmarks:
        decontam_texts = list(
            load_decontam_texts(
                sft.decontam_benchmarks,
                cleanup_hf_cache=sft.cleanup_hf_cache,
            )
        )
        log.info(
            "Decontam: collected %d benchmark strings from %d benchmark(s).",
            len(decontam_texts), len(sft.decontam_benchmarks),
        )

    # ── Dataset ────────────────────────────────────────────────────────────
    bundle = build_sft_datasets(
        hf_dataset_name=sft.hf_dataset_name,
        hf_split=sft.hf_split,
        tokenizer=tokenizer,
        template=DEFAULT_ALPACA_TEMPLATE,
        max_length=sft.max_length,
        val_fraction=sft.val_fraction,
        seed=42,
        decontam_texts=decontam_texts,
        max_examples=sft.max_examples,
        hf_cache_dir=sft.hf_cache_dir,
        cleanup_hf_cache=sft.cleanup_hf_cache,
        synthetic_jsonl_path=sft.synthetic_jsonl_path,
        synthetic_oversample=sft.synthetic_oversample,
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

    # ── Performance knobs (Blackwell sm_100 / Ada sm_89 / Ampere sm_8x) ────
    # TF32 matmul fallback gives free perf on FP32 paths (e.g. RoPE freqs
    # complex math, layer-norm reductions) without changing BF16 results.
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # `torch.compile` typically buys 1.4–1.8x on this size model. We use
    # the default mode (kernel fusion, no CUDA graphs). `reduce-overhead`
    # is faster but its CUDA-graph backend cannot handle our pattern of
    # calling the model in eval mode during validation, in train mode
    # during the loop, and a third time for the WT-103 tripwire — it
    # raises "accessing tensor output of CUDAGraphs that has been
    # overwritten by a subsequent run". Default mode avoids that.
    # The `_orig_mod` strip in _save_checkpoint already handles the
    # wrapper so checkpoints stay compatible with the un-compiled load.
    if sft.torch_compile and device.type == "cuda":
        log.info("Compiling model (torch.compile, mode='default')...")
        model = torch.compile(model, fullgraph=False)

    # ── Pretraining-mix setup (CF mitigation #1, VL07 slide 25) ────────────
    pretraining_memmap: np.ndarray | None = None
    mix_rng = np.random.default_rng(42)
    if sft.pretraining_mix_ratio > 0:
        if not sft.pretraining_bin_path:
            raise ValueError(
                "sft.pretraining_mix_ratio > 0 requires sft.pretraining_bin_path "
                "(path to the pretraining train.bin uint16 token array)."
            )
        if not Path(sft.pretraining_bin_path).exists():
            raise FileNotFoundError(
                f"sft.pretraining_bin_path does not exist: {sft.pretraining_bin_path}"
            )
        pretraining_memmap = np.memmap(sft.pretraining_bin_path, dtype=np.uint16, mode="r")
        log.info(
            "Pretraining-mix enabled: ratio=%.2f, source=%s (%d tokens)",
            sft.pretraining_mix_ratio, sft.pretraining_bin_path, len(pretraining_memmap),
        )

    # ── Wikitext-103 CF tripwire baseline (SFT.md §6) ─────────────────────
    # Compute PPL on the base checkpoint BEFORE any SFT step. Subsequent
    # tripwire evals compare against this baseline; if PPL rises >threshold,
    # the run hard-stops to preserve world knowledge.
    wt103_tokens: torch.Tensor | None = None
    wt103_baseline_ppl: float | None = None
    if sft.wt103_eval_every_n_evals > 0:
        log.info("Loading Wikitext-103 for catastrophic-forgetting tripwire...")
        wt103_tokens = _load_wt103_tokens(
            tokenizer,
            max_tokens=(sft.wt103_max_sequences + 2) * sft.max_length,
            cleanup_hf_cache=sft.cleanup_hf_cache,
        )
        model.eval()
        wt103_baseline_ppl = compute_perplexity(
            model, wt103_tokens, sft.max_length, device,
            batch_size=max(1, sft.batch_size),
            max_sequences=sft.wt103_max_sequences,
        )
        model.train()
        log.info("WT-103 baseline PPL on base model: %.3f", wt103_baseline_ppl)

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
    n_sup = _count_shifted_supervised_tokens(first["labels"])
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
    nonfinite_streak = 0
    oom_streak = 0
    should_stop_early = False

    for epoch in range(sft.epochs):
        state.epoch = epoch
        log.info("── Epoch %d/%d ──", epoch + 1, sft.epochs)
        for batch in train_loader:
            # Per-step pretraining-mix decision (VL07 slide 25 #1).
            # Replace this SFT batch with a pretraining batch with prob=ratio.
            use_mix = (
                pretraining_memmap is not None
                and float(mix_rng.random()) < sft.pretraining_mix_ratio
            )
            if use_mix:
                pre_inputs, pre_targets = _draw_pretraining_window(
                    pretraining_memmap, sft.max_length,
                    sft.batch_size, mix_rng,
                )
                pre_inputs = pre_inputs.to(device, non_blocking=True)
                pre_targets = pre_targets.to(device, non_blocking=True)
                with autocast:
                    _, loss = model(
                        pre_inputs,
                        targets=pre_targets,
                        return_logits=False,
                        loss_chunk_rows=sft.loss_chunk_rows,
                    )
                n_tokens = sft.max_length * sft.batch_size
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                n_tokens = _count_shifted_supervised_tokens(labels)
                if n_tokens == 0:
                    log.warning(
                        "Zero supervised next-token labels at step %d. "
                        "Skipping batch and resetting accumulator.",
                        state.step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    accum_loss_sum = 0.0
                    accum_tokens_sum = 0
                    oom_streak = 0
                    nonfinite_streak += 1
                    if nonfinite_streak >= 25:
                        raise RuntimeError(
                            "SFT saw 25 consecutive unusable/non-finite batches. "
                            "Stopping instead of looping forever; inspect the "
                            "label mask, response lengths, and optimizer health."
                        )
                    continue
                with autocast:
                    loss = model(
                        input_ids,
                        labels=labels,
                        loss_chunk_rows=sft.loss_chunk_rows,
                    )

            if _is_nonfinite(loss):
                # Discard the whole in-flight accumulation window — partial
                # gradients sitting in .grad from earlier batches would
                # otherwise be applied with the wrong divisor on the next
                # successful step.
                log.warning(
                    "Non-finite loss at step %d (NaN/Inf). Skipping batch and "
                    "resetting accumulator. Inspect the data pipeline if this "
                    "fires more than once per epoch.",
                    state.step,
                )
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                accum_loss_sum = 0.0
                accum_tokens_sum = 0
                _empty_device_cache(device)
                nonfinite_streak += 1
                if nonfinite_streak >= 25:
                    raise RuntimeError(
                        "SFT saw 25 consecutive unusable/non-finite batches. "
                        "Stopping instead of looping forever; inspect the "
                        "label mask, response lengths, and optimizer health."
                    )
                continue

            loss_value = loss.item()
            nonfinite_streak = 0

            # Gradient accumulation: divide the per-step loss by K so the
            # accumulated gradient has the correct magnitude after K backward
            # passes. This is VL04 slide 39 rule #3 ("Important: divide loss
            # by K, otherwise gradients are K times too large").
            loss_to_back = loss / sft.gradient_accumulation_steps
            try:
                if scaler is not None:
                    scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()
            except RuntimeError as exc:
                if not _is_recoverable_mps_oom(exc, device):
                    raise
                log.warning(
                    "MPS OOM during backward at step %d. Clearing device "
                    "cache, dropping the partial accumulation window, and "
                    "continuing with the next batch: %s",
                    state.step, exc,
                )
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                accum_loss_sum = 0.0
                accum_tokens_sum = 0
                oom_streak += 1
                del loss_to_back
                del loss
                _empty_device_cache(device)
                if oom_streak >= 5:
                    raise RuntimeError(
                        "SFT saw 5 consecutive recoverable MPS OOM windows. "
                        "Stopping instead of spinning on memory pressure. "
                        "Restart from the latest 'last' checkpoint after "
                        "lowering sft.max_length or closing memory-heavy apps."
                    ) from exc
                continue
            del loss_to_back
            if device.type == "mps":
                _empty_device_cache(device)
            oom_streak = 0

            # n_tokens was already computed above (branch-aware: response
            # tokens for SFT, full window for pretraining-mix).
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
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=sft.grad_clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    log.warning(
                        "Non-finite gradient norm at step %d. Skipping "
                        "optimizer step and resetting accumulator: %s",
                        state.step, exc,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    accum_loss_sum = 0.0
                    accum_tokens_sum = 0
                    _empty_device_cache(device)
                    nonfinite_streak += 1
                    if nonfinite_streak >= 25:
                        raise RuntimeError(
                            "SFT saw 25 consecutive non-finite gradient windows. "
                            "Stopping instead of looping forever; inspect the "
                            "loss graph and optimizer setup."
                        )
                    continue
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
            if not _model_parameters_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                _empty_device_cache(device)
                raise RuntimeError(
                    "Optimizer step produced NaN/Inf model parameters. "
                    "Aborting before the checkpoint is corrupted; on MPS this "
                    "usually means the optimizer kernel or learning-rate setup "
                    "became unstable."
                )
            optimizer.zero_grad(set_to_none=True)
            _empty_device_cache(device)

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
                val_metrics = _evaluate(
                    model,
                    val_loader,
                    device,
                    autocast,
                    loss_chunk_rows=sft.loss_chunk_rows,
                )
                _empty_device_cache(device)
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
                    state.bad_evals = 0
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
                else:
                    state.bad_evals += 1
                    if _should_stop_early(state.bad_evals, sft.early_stopping_patience):
                        log.info(
                            "Early stop: %d consecutive evals without improvement "
                            "(patience=%d). Best val loss = %.4f at %s.",
                            state.bad_evals, sft.early_stopping_patience,
                            state.best_val_loss, state.best_checkpoint,
                        )
                        should_stop_early = True

                # WT-103 catastrophic-forgetting tripwire (SFT.md §6).
                # Run every Nth SFT eval; compare against the base-model
                # baseline; hard-stop if PPL has risen above threshold.
                if (wt103_tokens is not None
                        and wt103_baseline_ppl is not None
                        and sft.wt103_eval_every_n_evals > 0):
                    eval_idx = state.step // sft.eval_every
                    if eval_idx % sft.wt103_eval_every_n_evals == 0:
                        cur_ppl = compute_perplexity(
                            model, wt103_tokens, sft.max_length, device,
                            batch_size=max(1, sft.batch_size),
                            max_sequences=sft.wt103_max_sequences,
                        )
                        delta_pct = 100.0 * (cur_ppl - wt103_baseline_ppl) / wt103_baseline_ppl
                        log.info(
                            "  [wt103] PPL %.3f (Δ %+.2f%% vs baseline %.3f)",
                            cur_ppl, delta_pct, wt103_baseline_ppl,
                        )
                        _write_metric({
                            "stage": "sft",
                            "step": state.step,
                            "epoch": epoch,
                            "wt103_ppl": cur_ppl,
                            "wt103_baseline_ppl": wt103_baseline_ppl,
                            "wt103_delta_pct": delta_pct,
                        })
                        if _wt103_should_stop(
                            baseline_ppl=wt103_baseline_ppl,
                            current_ppl=cur_ppl,
                            threshold_pct=sft.wt103_hard_stop_pct,
                        ):
                            log.warning(
                                "CF tripwire: WT-103 PPL up %.2f%% ≥ %.2f%% "
                                "hard-stop threshold. Halting SFT to preserve "
                                "world knowledge. Best SFT val: %.4f at %s.",
                                delta_pct, sft.wt103_hard_stop_pct,
                                state.best_val_loss, state.best_checkpoint,
                            )
                            should_stop_early = True

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

            if state.step >= total_optim_steps or should_stop_early:
                break

        # End of epoch: discard any partial accumulation. If we let the
        # leftover gradients carry into the next epoch, the first optim
        # step there would fire after fewer than K backward passes but
        # still divided by K, producing an under-magnitude update. The
        # cost of discarding (≤ K-1 batches per epoch boundary) is
        # negligible at our scale; the cost of a corrupted step is not.
        if accum_counter > 0:
            log.info(
                "Discarding %d partial-accumulation batches at end of epoch %d.",
                accum_counter, epoch + 1,
            )
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            accum_loss_sum = 0.0
            accum_tokens_sum = 0

        if state.step >= total_optim_steps or should_stop_early:
            break

    # Final checkpoint + summary.
    final_val = _evaluate(
        model,
        val_loader,
        device,
        autocast,
        loss_chunk_rows=sft.loss_chunk_rows,
    )
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
