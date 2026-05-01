"""Native PyTorch DPO trainer.

Mirrors the shape of EX08_DPO.ipynb's training loop but uses our model,
checkpointing, JSONL logger, and run-dir convention. The trainer runs two
forwards through the policy (chosen + rejected) and two through a frozen
reference model, then minimises the canonical DPO loss.

MPS guardrails (per the DPO Phase 1 spec, section 5):
- ``gradient_accumulation_steps`` must be 1 on MPS (autograd bug).
- Reference model is cast to bfloat16 on MPS to reduce memory pressure.
- Batch size > 4 emits a warning (two policy forwards + reference forward).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from configs import ProjectConfig
from src.logging_utils import JSONLLogger, make_run_dir
from src.model import ParrotLLM
from src.posttraining.dpo.loss import dpo_loss, sequence_logprob_from_labels
from src.training.trainer import (
    CheckpointManager,
    build_scheduler,
    get_autocast_context,
)


log = logging.getLogger("parrotllm.posttraining.dpo")


# ── Dataset / collator ──────────────────────────────────────────────────────


class PreferencePackedDataset(Dataset):
    """Reads packed preference records from a JSONL file produced by ``dpo-prepare``.

    Each record has the contract documented in
    ``src.posttraining.dpo.prepare``::

        {
          "prompt_tokens":   [...],
          "chosen_tokens":   [...prompt..., ...response_chosen...],
          "rejected_tokens": [...prompt..., ...response_rejected...],
          "prompt_len":      int,
        }
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


def _pad_to(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token sequences to the longest length.

    Returns (input_ids, attention_mask) as int64 / int64 tensors.
    """
    max_len = max(len(s) for s in seqs)
    batch_input = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    batch_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s)
        batch_input[i, :n] = torch.tensor(s, dtype=torch.long)
        batch_mask[i, :n] = 1
    return batch_input, batch_mask


def _labels_from(input_ids: torch.Tensor, prompt_lens: list[int],
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Build label tensors that mask prompt positions and padding with -100."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return labels


def build_dpo_collator(*, pad_token_id: int) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Return a collator that produces tensors keyed for the trainer.

    The returned batch dict has the keys::

        chosen_input_ids, chosen_labels, chosen_attention_mask,
        rejected_input_ids, rejected_labels, rejected_attention_mask,
        prompt_lens
    """

    def _collate(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen = [list(r["chosen_tokens"]) for r in records]
        rejected = [list(r["rejected_tokens"]) for r in records]
        prompt_lens = [int(r["prompt_len"]) for r in records]

        chosen_ids, chosen_mask = _pad_to(chosen, pad_token_id)
        rejected_ids, rejected_mask = _pad_to(rejected, pad_token_id)

        chosen_labels = _labels_from(chosen_ids, prompt_lens, chosen_mask)
        rejected_labels = _labels_from(rejected_ids, prompt_lens, rejected_mask)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_mask,
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        }

    return _collate


# ── Forward + train step ────────────────────────────────────────────────────


def _forward_logp(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log p(target tokens) per sequence, ignoring -100 positions.

    ``ParrotLLM.forward`` returns ``(logits, loss)``. We need only logits for
    DPO and pass ``targets=None`` to skip the internal CE computation.

    Causal-LM next-token alignment: predict label at position t from logits at
    position t-1, so we shift logits/labels by one before computing the
    sequence log-prob.
    """
    logits, _ = model(input_ids, targets=None, return_logits=True)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return sequence_logprob_from_labels(shift_logits, shift_labels)


def dpo_train_step(
    *,
    policy: nn.Module,
    reference: nn.Module,
    batch: dict[str, torch.Tensor],
    beta: float,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    autocast_ctx: Any | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one DPO optimizer step. Returns (loss, metrics_dict).

    ``metrics_dict`` always contains the float ``loss`` plus the six diagnostic
    keys returned by :func:`dpo_loss`.
    """
    policy.train()
    optimizer.zero_grad(set_to_none=True)

    chosen_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    rejected_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]

    if autocast_ctx is None:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    with autocast_ctx:
        pi_chosen = _forward_logp(policy, chosen_ids, chosen_labels)
        pi_rejected = _forward_logp(policy, rejected_ids, rejected_labels)

    # Reference forward never participates in the policy graph.
    with torch.no_grad():
        with autocast_ctx:
            ref_chosen = _forward_logp(reference, chosen_ids, chosen_labels)
            ref_rejected = _forward_logp(reference, rejected_ids, rejected_labels)
        ref_chosen = ref_chosen.detach().to(pi_chosen.dtype)
        ref_rejected = ref_rejected.detach().to(pi_rejected.dtype)

    loss, dpo_metrics = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=beta)
    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    metrics = {"loss": loss.detach().item()}
    metrics.update(dpo_metrics)
    return loss.detach(), metrics


# ── Eval helper ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _evaluate_chosen_token_accuracy(
    policy: nn.Module,
    reference: nn.Module,  # noqa: ARG001 — kept for symmetry with run_dpo
    dev_loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    """Return fraction of dev pairs where pi(chosen) > pi(rejected)."""
    policy.eval()
    n_total = 0
    n_correct = 0
    for batch in dev_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pi_c = _forward_logp(policy, batch["chosen_input_ids"], batch["chosen_labels"])
        pi_r = _forward_logp(policy, batch["rejected_input_ids"], batch["rejected_labels"])
        n_correct += (pi_c > pi_r).sum().item()
        n_total += pi_c.shape[0]
    policy.train()
    return n_correct / max(n_total, 1)


# ── Top-level run_dpo ────────────────────────────────────────────────────────


def run_dpo(project_config: ProjectConfig, *, device: torch.device, checkpoint: str | None = None) -> str:
    """Run DPO training end-to-end. Returns the run directory path."""
    dpo_cfg = project_config.dpo
    if dpo_cfg is None:
        raise ValueError("project config has no `dpo` section")
    if project_config.model is None:
        raise ValueError("project config has no `model` section")

    # MPS guardrails (per spec section 5).
    if device.type == "mps":
        if dpo_cfg.gradient_accumulation_steps != 1:
            raise ValueError(
                "On MPS, gradient_accumulation_steps must be 1 due to known autograd bug. "
                "Set dpo.gradient_accumulation_steps: 1 in the config."
            )
        if dpo_cfg.train_batch_size > 4:
            log.warning(
                "DPO on MPS with batch_size > 4 may OOM (two policy forwards + reference). "
                "Spec recommends batch_size <= 4."
            )

    # Reference model (frozen SFT checkpoint).
    ref_path = checkpoint or str(dpo_cfg.reference_checkpoint)
    log.info("Loading reference model from %s", ref_path)
    ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_state = ref_ckpt.get("model") or ref_ckpt.get("state_dict") or ref_ckpt
    project_dict = project_config.model_dump(mode="python")
    reference = ParrotLLM(project_dict)
    reference.load_state_dict(ref_state)
    reference = reference.to(device)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    if device.type == "mps":
        reference = reference.to(dtype=torch.bfloat16)  # spec section 5

    # Policy starts from the same weights.
    policy = ParrotLLM(project_dict)
    policy.load_state_dict(ref_state)
    policy = policy.to(device)

    # Data.
    train_path = Path(dpo_cfg.prepared_dir) / "train.jsonl"
    dev_path = Path(dpo_cfg.prepared_dir) / "dev.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"DPO data not found at {dpo_cfg.prepared_dir}. Run "
            f"`uv run python main.py --stage dpo-prepare --config <yaml>` first."
        )
    pad_id = int(project_config.model.pad_token_id)
    train_loader = DataLoader(
        PreferencePackedDataset(train_path),
        batch_size=dpo_cfg.train_batch_size,
        shuffle=True,
        collate_fn=build_dpo_collator(pad_token_id=pad_id),
    )
    dev_loader = DataLoader(
        PreferencePackedDataset(dev_path),
        batch_size=dpo_cfg.train_batch_size,
        shuffle=False,
        collate_fn=build_dpo_collator(pad_token_id=pad_id),
    )

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=dpo_cfg.learning_rate,
        betas=(dpo_cfg.beta1, dpo_cfg.beta2),
        weight_decay=dpo_cfg.weight_decay,
    )
    total_steps = max(1, int(dpo_cfg.num_epochs * len(train_loader)))
    warmup_steps = max(1, int(total_steps * dpo_cfg.warmup_ratio))
    min_lr = dpo_cfg.learning_rate * dpo_cfg.min_lr_ratio
    scheduler = build_scheduler(optimizer, {
        "warmup_steps": warmup_steps,
        "max_steps": total_steps,
        "min_lr": min_lr,
        "lr_schedule": "cosine",
        "lr_decay_ratio": dpo_cfg.min_lr_ratio,
    })

    # Run dir + logger + checkpointer (real helper signatures).
    run_dir = make_run_dir(str(dpo_cfg.runs_dir))                  # returns str
    ckpt_dir = str(Path(run_dir) / dpo_cfg.checkpoint_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    metrics_logger = JSONLLogger(run_dir)
    checkpointer = CheckpointManager(
        ckpt_dir,
        keep_last=dpo_cfg.keep_last_checkpoints,
        keep_best=dpo_cfg.keep_best_checkpoints,
    )

    # get_autocast_context returns (ctx, scaler); DPO doesn't use a GradScaler.
    autocast_ctx, _scaler_unused = get_autocast_context(device)

    # MPS allocator caches blocks aggressively and leaks slowly across long DPO
    # runs (4 forwards/step, 2 of which retain backward state). Periodically
    # returning the cache to the OS is the cheapest mitigation; without it the
    # 4688-step letter DPO run OOM'd at step 674. Frequency tuned empirically:
    # every 25 steps + after every eval keeps walltime overhead under ~3% while
    # bounding the cache below the per-process limit.
    def _empty_device_cache() -> None:
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    step = 0
    pbar = tqdm(total=total_steps, desc="dpo")
    done = False
    for epoch in range(int(math.ceil(dpo_cfg.num_epochs))):
        if done:
            break
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = dpo_train_step(
                policy=policy,
                reference=reference,
                batch=batch,
                beta=dpo_cfg.beta,
                optimizer=optimizer,
                grad_clip=dpo_cfg.grad_clip,
                autocast_ctx=autocast_ctx,
            )
            scheduler.step()
            step += 1
            pbar.update(1)

            if step % dpo_cfg.log_every == 0:
                lr = (
                    scheduler.get_last_lr()[0]
                    if hasattr(scheduler, "get_last_lr")
                    else dpo_cfg.learning_rate
                )
                metrics_logger.log(
                    "dpo", "train",
                    step=step,
                    epoch=epoch,
                    lr=lr,
                    **metrics,
                )

            # Periodic cache release — bounds MPS allocator growth.
            if step % 25 == 0:
                _empty_device_cache()

            if step % dpo_cfg.eval_every == 0:
                acc = _evaluate_chosen_token_accuracy(
                    policy, reference, dev_loader, device=device,
                )
                metrics_logger.log(
                    "dpo", "eval",
                    step=step,
                    epoch=epoch,
                    chosen_token_accuracy=acc,
                )
                # CheckpointManager.maybe_save_best uses LOWER-IS-BETTER val_loss;
                # convert accuracy (higher-is-better) into 1 - acc.
                checkpointer.maybe_save_best(
                    policy, optimizer,
                    project_config.model_dump(mode="python"),
                    step, epoch,
                    None,  # no GradScaler in DPO
                    val_loss=1.0 - acc,
                )
                # Eval traverses the full dev set (2 forwards/example) and is a
                # known leak hotspot — drop the cache once the eval is logged.
                _empty_device_cache()

            if step % dpo_cfg.save_every == 0:
                checkpointer.save_last(
                    policy, optimizer,
                    project_config.model_dump(mode="python"),
                    step, epoch,
                    None,  # no GradScaler in DPO
                )

            if step >= total_steps:
                done = True
                break
    pbar.close()
    metrics_logger.close()
    log.info("DPO training complete after %d steps", step)
    return run_dir


__all__ = [
    "PreferencePackedDataset",
    "build_dpo_collator",
    "dpo_train_step",
    "run_dpo",
]
