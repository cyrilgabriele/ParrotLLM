"""Training loop for supervised fine-tuning (SFT)."""

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from configs import ProjectConfig
from src.logging_utils import JSONLLogger, make_run_dir, setup_logger
from src.model import ParrotLLM
from src.posttraining.eval import evaluate_prompt_suite
from src.training.trainer import (
    CheckpointManager,
    WindowDataset,
    _empty_device_cache,
    build_optimizer,
    build_scheduler,
    estimate_loss,
    get_autocast_context,
    resolve_checkpoint_dir,
)
from src.utils import build_tokenizer


log = logging.getLogger("parrotllm.posttraining")


class PackedConversationDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def build_sft_collator(pad_token_id: int):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        seq_len = max(len(item["tokens"]) for item in batch) - 1
        batch_size = len(batch)
        x = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
        y = torch.zeros((batch_size, seq_len), dtype=torch.long)
        loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)
        for row_idx, item in enumerate(batch):
            tokens = item["tokens"]
            mask = item["loss_mask"]
            n = len(tokens) - 1
            x[row_idx, :n] = torch.tensor(tokens[:-1], dtype=torch.long)
            y[row_idx, :n] = torch.tensor(tokens[1:], dtype=torch.long)
            loss_mask[row_idx, :n] = torch.tensor(mask[1:], dtype=torch.float32)
        return {"x": x, "y": y, "loss_mask": loss_mask}

    return _collate


def _build_quality_sampler(
    dataset: PackedConversationDataset,
    *,
    seed: int,
    num_samples: int | None = None,
) -> torch.utils.data.WeightedRandomSampler:
    """Sample packed records proportional to their `quality_score` field.

    Records without quality_score default to weight 1.0. Quality routing
    happens at the data-mixing level, not via per-example loss weighting,
    matching canonical SFT practice (LIMA, Tulu 2/3, Alpaca).
    """
    weights = torch.tensor(
        [float(rec.get("quality_score", 1.0)) for rec in dataset.records],
        dtype=torch.double,
    )
    weights = weights.clamp_min(1e-6)
    n = num_samples if num_samples is not None else len(dataset)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=n,
        replacement=True,
        generator=generator,
    )


@dataclass(slots=True)
class SweepResult:
    learning_rate: float
    run_dir: str
    best_checkpoint: str | None
    best_score: float
    best_dev_loss: float
    best_replay_ppl: float
    best_format_score: float


def _seed_worker(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    z_loss_coeff: float = 0.0,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    weighted = losses * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    loss = weighted.sum() / denom
    if z_loss_coeff > 0.0:
        z_term = torch.logsumexp(logits.float(), dim=-1).pow(2)
        loss = loss + (z_term * loss_mask).sum() / denom * z_loss_coeff
    return loss


def _loss_chunk_rows_for_device(device: torch.device) -> int:
    if device.type == "mps":
        return 16
    return 2048


def _resolve_runtime_batching(
    *,
    device: torch.device,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int, int]:
    if device.type != "mps":
        return train_batch_size, eval_batch_size, gradient_accumulation_steps

    # On Apple MPS, memory drift is a real issue. Batch 1 is the most stable.
    # We compensate for the small batch size by increasing gradient accumulation.
    safe_train_batch_size = min(train_batch_size, 1)
    safe_eval_batch_size = min(eval_batch_size, 2)

    if safe_train_batch_size == train_batch_size:
        return train_batch_size, safe_eval_batch_size, gradient_accumulation_steps

    effective_sequences = train_batch_size * gradient_accumulation_steps
    safe_gradient_accumulation_steps = max(
        gradient_accumulation_steps,
        math.ceil(effective_sequences / safe_train_batch_size),
    )
    return safe_train_batch_size, safe_eval_batch_size, safe_gradient_accumulation_steps


def _make_replay_loader(
    path: Path,
    *,
    context_length: int,
    batch_size: int,
    seed: int,
) -> DataLoader | None:
    if not path.exists():
        return None
    dataset = WindowDataset(str(path), context_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def _cycle(loader: DataLoader | None):
    if loader is None:
        while True:
            yield None
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _evaluate_sft_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    autocast_ctx,
    z_loss_coeff: float,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total = 0.0
    denom = 0.0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        with autocast_ctx:
            _, loss = model(
                x,
                targets=y,
                loss_mask=loss_mask,
                return_logits=False,
                z_loss_coeff=z_loss_coeff,
                loss_chunk_rows=_loss_chunk_rows_for_device(device),
            )
        total += float(loss.detach().item()) * float(loss_mask.sum().item())
        denom += float(loss_mask.sum().item())
    model.train()
    if denom <= 0:
        return float("inf")
    return total / denom


def _load_manifest(prepared_dir: Path) -> dict[str, Any]:
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared SFT manifest not found at {manifest_path}. Run --stage sft-prepare first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _count_jsonl_records(path: str | Path | None) -> int:
    if path is None:
        return 0
    record_path = Path(path)
    if not record_path.exists():
        return 0
    with record_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _merge_model_config(project_config: ProjectConfig, checkpoint_config: dict[str, Any] | None) -> dict[str, Any]:
    payload = project_config.model_dump(mode="python")
    if checkpoint_config and "model" in checkpoint_config:
        payload["model"] = checkpoint_config["model"]
    return payload


def _load_base_model(
    project_config: ProjectConfig,
    *,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = checkpoint.get("config", {})
    merged_config = _merge_model_config(project_config, ckpt_config)
    model = ParrotLLM(merged_config).to(device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint  # Explicitly release checkpoint memory
    gc.collect()
    model.train()
    return model, merged_config


def _evaluate_checkpoint(
    *,
    model: torch.nn.Module,
    dev_loader: DataLoader,
    replay_val_ds,
    sft_cfg,
    device: torch.device,
    autocast_ctx,
    tokenizer,
    context_length: int,
    generation_log_label: str | None = None,
) -> dict[str, float]:
    dev_loss = _evaluate_sft_loss(
        model,
        dev_loader,
        device=device,
        autocast_ctx=autocast_ctx,
        z_loss_coeff=sft_cfg.z_loss_coeff,
    )
    metrics = {"dev_loss": float(dev_loss), "replay_ppl": float("inf"), "format_score": 0.0}
    if replay_val_ds is not None:
        replay_metrics = estimate_loss(
            model,
            replay_val_ds,
            device,
            autocast_ctx,
            sft_cfg.eval_batch_size,
            max_batches=20,
        )
        metrics["replay_ppl"] = float(replay_metrics["perplexity"])
    suite = evaluate_prompt_suite(
        model,
        tokenizer,
        path=sft_cfg.prompt_suite_path,
        device=device,
        system_prompt=sft_cfg.system_prompt,
        context_length=context_length,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        return_generations=bool(sft_cfg.log_prompt_suite_generations),
    )
    metrics["format_score"] = float(suite.get("format_score", 0.0))
    if sft_cfg.log_prompt_suite_generations:
        _log_prompt_suite_generations(
            suite.get("generations", []),
            label=generation_log_label or "eval",
            format_score=metrics["format_score"],
        )
    if device.type == "mps":
        gc.collect()
        _empty_device_cache(device)
    return metrics


def _terminal_bar(value: float, *, width: int = 12) -> str:
    filled = max(0, min(width, round(float(value) * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _one_line(text: Any, *, limit: int = 140) -> str:
    normalized = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _log_prompt_suite_generations(
    generations: Any,
    *,
    label: str,
    format_score: float,
) -> None:
    if not isinstance(generations, list) or not generations:
        return
    lines = [
        "",
        f"Prompt-suite generation check ({label}) "
        f"format_score={format_score:.3f} {_terminal_bar(format_score)}",
    ]
    for record in generations:
        if not isinstance(record, dict):
            continue
        score = float(record.get("score", 0.0))
        idx = int(record.get("case_index", 0))
        response = _one_line(record.get("response", ""))
        raw_generated = _one_line(record.get("raw_generated", ""))
        shown = response or raw_generated or "<empty>"
        lines.append(f"  case {idx:02d} score={score:.1f} {_terminal_bar(score, width=8)} {shown}")
    log.info("\n".join(lines))


def _composite_score(
    metrics: dict[str, float],
    *,
    base_replay_ppl: float | None,
    format_weight: float,
    forgetting_weight: float,
) -> float:
    score = float(metrics["dev_loss"])
    score += format_weight * (1.0 - float(metrics.get("format_score", 0.0)))
    if base_replay_ppl is not None and math.isfinite(base_replay_ppl):
        replay_penalty = max(0.0, float(metrics.get("replay_ppl", base_replay_ppl)) - base_replay_ppl)
        score += forgetting_weight * replay_penalty
    return score


def _early_stopping_value(metrics: dict[str, float], composite_score: float, metric: str) -> float:
    if metric == "composite_score":
        return float(composite_score)
    if metric not in metrics:
        raise ValueError(
            f"Unsupported early_stopping_metric '{metric}'. "
            f"Available metrics: composite_score, {', '.join(sorted(metrics))}"
        )
    return float(metrics[metric])


def _early_stopping_improved(current: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return current > (best + min_delta)
    return current < (best - min_delta)


def _early_stopping_target_reached(current: float, target: float | None, *, mode: str) -> bool:
    if target is None:
        return False
    if mode == "max":
        return current >= target
    return current <= target


def _run_single_sweep(
    project_config: ProjectConfig,
    *,
    device: torch.device,
    base_checkpoint: str,
    prepared_dir: Path,
    learning_rate: float,
    train_dataset_path: Path,
    dev_dataset_path: Path,
    tag: str,
) -> SweepResult:
    # Disable MPS allocation cap to utilize more shared memory if available
    if device.type == "mps":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    sft_cfg = project_config.sft
    assert sft_cfg is not None

    manifest = _load_manifest(prepared_dir)
    tokenizer = build_tokenizer()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad token for SFT batching.")

    train_batch_size, eval_batch_size, gradient_accumulation_steps = _resolve_runtime_batching(
        device=device,
        train_batch_size=int(sft_cfg.train_batch_size),
        eval_batch_size=int(sft_cfg.eval_batch_size),
        gradient_accumulation_steps=int(sft_cfg.gradient_accumulation_steps),
    )
    if (
        train_batch_size != int(sft_cfg.train_batch_size)
        or eval_batch_size != int(sft_cfg.eval_batch_size)
        or gradient_accumulation_steps != int(sft_cfg.gradient_accumulation_steps)
    ):
        log.warning(
            "MPS runtime batching override for %s: train_batch_size %d -> %d, "
            "eval_batch_size %d -> %d, gradient_accumulation_steps %d -> %d",
            tag,
            int(sft_cfg.train_batch_size),
            train_batch_size,
            int(sft_cfg.eval_batch_size),
            eval_batch_size,
            int(sft_cfg.gradient_accumulation_steps),
            gradient_accumulation_steps,
        )

    train_ds = PackedConversationDataset(train_dataset_path)
    dev_ds = PackedConversationDataset(dev_dataset_path)
    collate_fn = build_sft_collator(pad_token_id)

    sampler = _build_quality_sampler(
        train_ds,
        seed=int(sft_cfg.seed),
        num_samples=len(train_ds),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model, effective_config = _load_base_model(
        project_config,
        checkpoint_path=base_checkpoint,
        device=device,
    )
    _empty_device_cache(device)
    context_length = int(effective_config["model"]["context_length"])

    if device.type == "cuda" and sft_cfg.compile:
        model = torch.compile(model)

    tc = {
        "learning_rate": learning_rate,
        "weight_decay": sft_cfg.weight_decay,
        "beta1": sft_cfg.beta1,
        "beta2": sft_cfg.beta2,
        "warmup_steps": 0,
        "max_steps": 1,
        "min_lr": learning_rate * sft_cfg.min_lr_ratio,
        "lr_schedule": "cosine",
        "lr_decay_ratio": 1.0,
    }

    micro_batches = max(1, math.ceil(len(train_loader) * sft_cfg.num_epochs))
    optimizer_steps = max(1, math.ceil(micro_batches / gradient_accumulation_steps))
    tc["max_steps"] = optimizer_steps
    tc["warmup_steps"] = int(round(optimizer_steps * sft_cfg.warmup_ratio))

    optimizer = build_optimizer(model, tc)
    scheduler = build_scheduler(optimizer, tc)
    autocast_ctx, scaler = get_autocast_context(device)

    replay_train_loader = None
    if sft_cfg.replay_ratio > 0.0:
        replay_train_loader = _make_replay_loader(
            Path(sft_cfg.replay_train_bin),
            context_length=context_length,
            batch_size=train_batch_size,
            seed=int(sft_cfg.seed),
        )
    replay_train_iter = _cycle(replay_train_loader)
    replay_val_ds = None
    if sft_cfg.replay_ratio > 0.0 and Path(sft_cfg.replay_val_bin).exists():
        from src.training.trainer import StridedWindowDataset

        replay_val_ds = StridedWindowDataset(str(sft_cfg.replay_val_bin), context_length, stride=context_length // 2)

    run_dir = make_run_dir(str(sft_cfg.runs_dir), tag=tag)
    logging_cfg = project_config.logging
    setup_logger(
        run_dir,
        console_level=logging_cfg.console_level if logging_cfg else "INFO",
        file_level=logging_cfg.file_level if logging_cfg else "DEBUG",
        component_levels=logging_cfg.components if logging_cfg and logging_cfg.components else None,
    )
    jlog = JSONLLogger(run_dir)
    checkpoint_dir = resolve_checkpoint_dir(run_dir, sft_cfg.checkpoint_dir)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        keep_last=sft_cfg.keep_last_checkpoints,
        keep_best=sft_cfg.keep_best_checkpoints,
    )

    (Path(run_dir) / "config.json").write_text(
        json.dumps(project_config.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    (Path(run_dir) / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loss_chunk_rows = _loss_chunk_rows_for_device(device)
    log.info(
        "Starting %s on %s with train_batch_size=%d eval_batch_size=%d grad_accum=%d optimizer_steps=%d loss_chunk_rows=%d",
        tag,
        device,
        train_batch_size,
        eval_batch_size,
        gradient_accumulation_steps,
        optimizer_steps,
        loss_chunk_rows,
    )
    prompt_suite_cases = _count_jsonl_records(sft_cfg.prompt_suite_path)
    log.info(
        "Initial checkpoint evaluation before optimizer step 1: %d dev batches, up to %d replay batches, %d prompt-suite cases. "
        "On %s the visible training bar advances per micro-batch, while optimizer steps happen every %d micro-batches.",
        len(dev_loader),
        20 if replay_val_ds is not None else 0,
        prompt_suite_cases,
        device,
        gradient_accumulation_steps,
    )

    base_metrics = _evaluate_checkpoint(
        model=model,
        dev_loader=dev_loader,
        replay_val_ds=replay_val_ds,
        sft_cfg=sft_cfg,
        device=device,
        autocast_ctx=autocast_ctx,
        tokenizer=tokenizer,
        context_length=context_length,
        generation_log_label="step 0",
    )
    _empty_device_cache(device)
    base_replay_ppl = base_metrics["replay_ppl"] if math.isfinite(base_metrics["replay_ppl"]) else None
    best_score = _composite_score(
        base_metrics,
        base_replay_ppl=base_replay_ppl,
        format_weight=sft_cfg.format_score_weight,
        forgetting_weight=sft_cfg.forgetting_penalty_weight,
    )
    best_early_stop_value = _early_stopping_value(
        base_metrics,
        best_score,
        sft_cfg.early_stopping_metric,
    )
    best_metrics = dict(base_metrics)
    early_stop_bad_evals = 0
    best_checkpoint = checkpoint_manager.maybe_save_best(
        model,
        optimizer,
        effective_config,
        step=0,
        epoch=0,
        scaler=scaler,
        val_loss=best_score,
        scheduler=scheduler,
        trainer_state={"selection_metric": "composite_score"},
    )

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    rng = random.Random(int(sft_cfg.seed))
    progress = tqdm(
        total=micro_batches,
        desc=tag,
        leave=True,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    while micro_step < micro_batches:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        use_replay = replay_train_loader is not None and rng.random() < sft_cfg.replay_ratio
        if use_replay:
            replay_batch = next(replay_train_iter)
            x = replay_batch[0].to(device)
            y = replay_batch[1].to(device)
            with autocast_ctx:
                _, loss = model(
                    x,
                    targets=y,
                    return_logits=False,
                    loss_chunk_rows=loss_chunk_rows,
                )
        else:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            with autocast_ctx:
                _, loss = model(
                    x,
                    y,
                    loss_mask=loss_mask,
                    return_logits=False,
                    z_loss_coeff=sft_cfg.z_loss_coeff,
                    loss_chunk_rows=loss_chunk_rows,
                )

        scaled_loss = loss / gradient_accumulation_steps
        if scaler is None:
            scaled_loss.backward()
        else:
            scaler.scale(scaled_loss).backward()

        micro_step += 1
        progress.update(1)
        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.3f}",
            opt=f"{step}/{optimizer_steps}",
            refresh=False,
        )
        if micro_step % gradient_accumulation_steps != 0 and micro_step < micro_batches:
            continue

        if scaler is None:
            if sft_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            if sft_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        # Periodic memory cleanup on MPS
        if device.type == "mps":
            gc.collect()
            _empty_device_cache(device)

        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.3f}",
            opt=f"{step}/{optimizer_steps}",
            lr=f"{float(optimizer.param_groups[0]['lr']):.2e}",
            refresh=False,
        )

        if step % sft_cfg.log_every == 0:
            jlog.log(
                "sft",
                "step",
                step=step,
                micro_step=micro_step,
                train_loss=float(loss.detach().item()),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )

        if step % sft_cfg.save_every == 0:
            checkpoint_manager.save_last(
                model,
                optimizer,
                effective_config,
                step=step,
                epoch=0,
                scaler=scaler,
                scheduler=scheduler,
                trainer_state={"selection_metric": "composite_score"},
            )

        should_eval = step % sft_cfg.eval_every == 0 or step == optimizer_steps
        if should_eval:
            if device.type == "mps":
                gc.collect()
                _empty_device_cache(device)
            metrics = _evaluate_checkpoint(
                model=model,
                dev_loader=dev_loader,
                replay_val_ds=replay_val_ds,
                sft_cfg=sft_cfg,
                device=device,
                autocast_ctx=autocast_ctx,
                tokenizer=tokenizer,
                context_length=context_length,
                generation_log_label=f"step {step}",
            )
            if device.type == "mps":
                gc.collect()
                _empty_device_cache(device)
            score = _composite_score(
                metrics,
                base_replay_ppl=base_replay_ppl,
                format_weight=sft_cfg.format_score_weight,
                forgetting_weight=sft_cfg.forgetting_penalty_weight,
            )
            early_stop_value = _early_stopping_value(
                metrics,
                score,
                sft_cfg.early_stopping_metric,
            )
            jlog.log(
                "sft",
                "eval",
                step=step,
                dev_loss=metrics["dev_loss"],
                replay_ppl=metrics["replay_ppl"],
                format_score=metrics["format_score"],
                composite_score=score,
                early_stopping_metric=sft_cfg.early_stopping_metric,
                early_stopping_value=early_stop_value,
            )
            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.3f}",
                opt=f"{step}/{optimizer_steps}",
                dev=f"{float(metrics['dev_loss']):.3f}",
                score=f"{float(score):.3f}",
                refresh=False,
            )
            improved = score < (best_score - sft_cfg.early_stopping_min_delta)
            if improved:
                best_score = score
                best_metrics = metrics
                best_checkpoint = checkpoint_manager.maybe_save_best(
                    model,
                    optimizer,
                    effective_config,
                    step=step,
                    epoch=0,
                    scaler=scaler,
                    val_loss=score,
                    scheduler=scheduler,
                    trainer_state={"selection_metric": "composite_score"},
                )

            early_stop_improved = _early_stopping_improved(
                early_stop_value,
                best_early_stop_value,
                mode=sft_cfg.early_stopping_mode,
                min_delta=sft_cfg.early_stopping_min_delta,
            )
            if early_stop_improved:
                best_early_stop_value = early_stop_value
                early_stop_bad_evals = 0
            elif sft_cfg.early_stopping_patience > 0:
                early_stop_bad_evals += 1
                log.info(
                    "SFT early stopping: no %s improvement for %d/%d evals "
                    "(current=%.4f best=%.4f mode=%s min_delta=%.4g).",
                    sft_cfg.early_stopping_metric,
                    early_stop_bad_evals,
                    sft_cfg.early_stopping_patience,
                    early_stop_value,
                    best_early_stop_value,
                    sft_cfg.early_stopping_mode,
                    sft_cfg.early_stopping_min_delta,
                )
                jlog.log(
                    "sft",
                    "early_stopping",
                    step=step,
                    metric=sft_cfg.early_stopping_metric,
                    mode=sft_cfg.early_stopping_mode,
                    bad_evals=early_stop_bad_evals,
                    patience=sft_cfg.early_stopping_patience,
                    best_value=best_early_stop_value,
                    current_value=early_stop_value,
                    min_delta=sft_cfg.early_stopping_min_delta,
                )
                if early_stop_bad_evals >= sft_cfg.early_stopping_patience:
                    log.info(
                        "Stopping SFT early at step %d/%d after %d non-improving evals.",
                        step,
                        optimizer_steps,
                        early_stop_bad_evals,
                    )
                    break
            if _early_stopping_target_reached(
                early_stop_value,
                sft_cfg.early_stopping_target,
                mode=sft_cfg.early_stopping_mode,
            ):
                log.info(
                    "Stopping SFT early at step %d/%d because %s reached target %.4f "
                    "(current=%.4f).",
                    step,
                    optimizer_steps,
                    sft_cfg.early_stopping_metric,
                    sft_cfg.early_stopping_target,
                    early_stop_value,
                )
                jlog.log(
                    "sft",
                    "early_stopping_target",
                    step=step,
                    metric=sft_cfg.early_stopping_metric,
                    mode=sft_cfg.early_stopping_mode,
                    target=sft_cfg.early_stopping_target,
                    current_value=early_stop_value,
                )
                break

    progress.close()
    checkpoint_manager.save_last(
        model,
        optimizer,
        effective_config,
        step=step,
        epoch=0,
        scaler=scaler,
        scheduler=scheduler,
        trainer_state={"selection_metric": "composite_score"},
    )
    jlog.close()
    
    # Explicitly clear model and optimizer before returning to release memory
    del model, optimizer, scheduler, train_loader, dev_loader
    if device.type == "mps":
        gc.collect()
        _empty_device_cache(device)

    return SweepResult(
        learning_rate=learning_rate,
        run_dir=run_dir,
        best_checkpoint=best_checkpoint,
        best_score=best_score,
        best_dev_loss=float(best_metrics["dev_loss"]),
        best_replay_ppl=float(best_metrics["replay_ppl"]),
        best_format_score=float(best_metrics["format_score"]),
    )


def _select_polish_subset(prepared_dir: Path, subset_size: int) -> Path:
    train_examples_path = prepared_dir / "train_examples.jsonl"
    records = PackedConversationDataset(train_examples_path).records
    records.sort(
        key=lambda item: (
            -float(item.get("quality_score", 0.0)),
            item.get("prompt_hash", ""),
        )
    )
    subset = records[:subset_size]
    output_path = prepared_dir / "polish_examples.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in subset:
            handle.write(json.dumps(record) + "\n")
    return output_path


def _pack_polish_examples(source_path: Path, output_path: Path, *, max_seq_length: int) -> Path:
    from src.posttraining.prepare import PreparedExample, _pack_examples

    items: list[PreparedExample] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            items.append(PreparedExample(**payload))
    packed = _pack_examples(items, max_seq_length=max_seq_length)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in packed:
            handle.write(json.dumps(record) + "\n")
    return output_path


def run_sft(
    project_config: ProjectConfig,
    *,
    device: torch.device,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    sft_cfg = project_config.sft
    if sft_cfg is None:
        raise ValueError("SFT configuration missing; cannot run SFT.")

    prepared_dir = Path(sft_cfg.prepared_dir)
    manifest = _load_manifest(prepared_dir)
    base_checkpoint = checkpoint or str(sft_cfg.base_checkpoint)
    if not Path(base_checkpoint).exists():
        raise FileNotFoundError(
            f"Base checkpoint not found at {base_checkpoint}. "
            "Place it under runs/posttraining/base_import/run_20260422_manual/checkpoints/base_pretrain.pt "
            "or pass --checkpoint."
        )

    train_dataset_path = Path(manifest["train_packed_path"])
    dev_dataset_path = Path(manifest["dev_packed_path"])

    sweep_results: list[SweepResult] = []
    for lr in sft_cfg.learning_rates:
        tag = f"sft_lr_{str(lr).replace('.', 'p')}"
        result = _run_single_sweep(
            project_config,
            device=device,
            base_checkpoint=base_checkpoint,
            prepared_dir=prepared_dir,
            learning_rate=float(lr),
            train_dataset_path=train_dataset_path,
            dev_dataset_path=dev_dataset_path,
            tag=tag,
        )
        sweep_results.append(result)

    best_sweep = min(sweep_results, key=lambda item: item.best_score)

    polish_summary: dict[str, Any]
    if sft_cfg.polish_epochs > 0.0:
        polish_source = _select_polish_subset(prepared_dir, sft_cfg.polish_subset_size)
        polish_packed = _pack_polish_examples(
            polish_source,
            prepared_dir / "polish_packed.jsonl",
            max_seq_length=sft_cfg.max_seq_length,
        )
        polish_project_config = project_config.model_copy(deep=True)
        assert polish_project_config.sft is not None
        polish_project_config.sft.num_epochs = sft_cfg.polish_epochs
        polish_project_config.sft.learning_rates = [best_sweep.learning_rate / 2.0]
        polish_result = _run_single_sweep(
            polish_project_config,
            device=device,
            base_checkpoint=best_sweep.best_checkpoint or base_checkpoint,
            prepared_dir=prepared_dir,
            learning_rate=best_sweep.learning_rate / 2.0,
            train_dataset_path=polish_packed,
            dev_dataset_path=dev_dataset_path,
            tag="sft_polish",
        )
        polish_summary = {
            "learning_rate": polish_result.learning_rate,
            "run_dir": polish_result.run_dir,
            "best_checkpoint": polish_result.best_checkpoint,
            "best_score": polish_result.best_score,
            "best_dev_loss": polish_result.best_dev_loss,
            "best_replay_ppl": polish_result.best_replay_ppl,
            "best_format_score": polish_result.best_format_score,
        }
    else:
        log.info("Skipping polish pass because polish_epochs=%s", sft_cfg.polish_epochs)
        polish_summary = {
            "skipped": True,
            "reason": "polish_epochs <= 0",
        }

    summary = {
        "base_checkpoint": base_checkpoint,
        "best_sweep": {
            "learning_rate": best_sweep.learning_rate,
            "run_dir": best_sweep.run_dir,
            "best_checkpoint": best_sweep.best_checkpoint,
            "best_score": best_sweep.best_score,
            "best_dev_loss": best_sweep.best_dev_loss,
            "best_replay_ppl": best_sweep.best_replay_ppl,
            "best_format_score": best_sweep.best_format_score,
        },
        "polish": polish_summary,
        "sweeps": [
            {
                "learning_rate": result.learning_rate,
                "run_dir": result.run_dir,
                "best_checkpoint": result.best_checkpoint,
                "best_score": result.best_score,
                "best_dev_loss": result.best_dev_loss,
                "best_replay_ppl": result.best_replay_ppl,
                "best_format_score": result.best_format_score,
            }
            for result in sweep_results
        ],
    }
    summary_path = prepared_dir / "sft_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("SFT summary written to %s", summary_path)
    return summary
