"""Training loop for ParrotLLM pretraining."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

import numpy as np
import torch
import torch.nn as nn

from configs import ProjectConfig
from src.logging_utils import (
    JSONLLogger, fmt_model_summary, fmt_training_complete,
    fmt_training_start, make_run_dir, render_ascii_loss_curve, setup_logger,
)
from src.model import ParrotLLM


# ── Dataset ──────────────────────────────────────────────────────────────────

class PretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path: str, context_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        # drop the last partial chunk
        self.n_chunks = len(self.data) // (context_length + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.context_length + 1)
        chunk = torch.from_numpy(
            self.data[start : start + self.context_length + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


# ── LR Schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Optimizer ────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, tc: dict) -> torch.optim.AdamW:
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": tc["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=tc["learning_rate"],
        betas=(tc["beta1"], tc["beta2"]),
        fused=torch.cuda.is_available(),
    )


# ── Mixed Precision helpers ──────────────────────────────────────────────────

def get_autocast_context(device: torch.device):
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:  # Ampere+
            return torch.autocast("cuda", dtype=torch.bfloat16), None
        else:
            scaler = torch.amp.GradScaler("cuda")
            return torch.autocast("cuda", dtype=torch.float16), scaler
    return torch.autocast(device.type, enabled=False), None


# ── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    config: dict, step: int, epoch: int,
                    scaler: torch.amp.GradScaler | None,
                    checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{epoch:02d}_epoch_{step}_step")
    # unwrap compiled model if needed
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    logging.getLogger("parrotllm.training").debug(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scaler: torch.amp.GradScaler | None = None,
                    device: torch.device = torch.device("cpu")):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("config", {})


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model: nn.Module, dataset: PretrainingDataset,
                  device: torch.device, autocast_ctx, batch_size: int,
                  max_batches: int = 20) -> dict:
    model.eval()
    losses = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
    )
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses) if losses else float("nan")
    return {"loss": avg, "perplexity": math.exp(avg) if avg == avg else float("nan")}


# ── Training ─────────────────────────────────────────────────────────────────

def _log_model_architecture(log: logging.Logger, jlog: JSONLLogger,
                            model: nn.Module, mc: dict,
                            device: torch.device, batch_size: int) -> None:
    """Log model architecture summary."""
    # Deduplicate by data_ptr to handle weight tying (tok_emb == lm_head)
    seen = set()
    n_total = 0
    n_trainable = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            n_total += p.numel()
            if p.requires_grad:
                n_trainable += p.numel()
    n_non_trainable = n_total - n_trainable
    pos_emb_params = model.pos_emb.weight.numel()
    n_non_emb = n_total - pos_emb_params
    params_size_mb = n_total * 4 / 1e6

    # torchinfo summary (if available)
    torchinfo_str = None
    try:
        from torchinfo import summary
        dummy_input = torch.randint(
            0, mc["vocab_size"], (batch_size, mc["context_length"]), device=device,
        )
        stats = summary(
            model, input_data=(dummy_input,),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            depth=3, verbose=0,
        )
        torchinfo_str = str(stats)
    except ImportError:
        log.debug("torchinfo not installed, skipping layer-wise summary")

    log.info(fmt_model_summary(
        mc,
        n_params=n_total, n_non_emb=n_non_emb, pos_emb_params=pos_emb_params,
        n_trainable=n_trainable, n_non_trainable=n_non_trainable,
        params_size_mb=params_size_mb, torchinfo=torchinfo_str,
    ))

    jlog.log("pretraining", "model_architecture",
             vocab_size=mc["vocab_size"],
             context_length=mc["context_length"],
             n_layers=mc["n_layers"], n_heads=mc["n_heads"],
             d_model=mc["d_model"], d_ff=mc["d_ff"],
             dropout=mc.get("dropout", 0.0), bias=mc.get("bias", False),
             total_params=n_total, total_params_non_embedding=n_non_emb,
             trainable_params=n_trainable, non_trainable_params=n_non_trainable,
             params_size_mb=round(params_size_mb, 2))


def run_train(
    project_config: ProjectConfig,
    model_config_dict: dict,
    *,
    device: torch.device,
    checkpoint: str | None = None,
    trial: optuna.Trial | None = None,
) -> float:
    """Train ParrotLLM using a fully validated project configuration."""

    tc_model = project_config.training
    mc_model = project_config.model
    lc_model = project_config.logging

    tc = tc_model.model_dump()
    mc = mc_model.model_dump()

    # ── run directory & loggers ───────────────────────────────────────────────
    run_dir = make_run_dir(tc_model.runs_dir)
    setup_logger(
        run_dir,
        console_level=lc_model.console_level,
        file_level=lc_model.file_level,
        component_levels=lc_model.components if lc_model.components else None,
    )
    log = logging.getLogger("parrotllm.training")
    jlog = JSONLLogger(run_dir)

    # Save full config to run directory for reproducibility
    config_path = os.path.join(run_dir, "config.json")
    json_payload = project_config.model_dump(mode="json")
    with open(config_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    jlog.log("pretraining", "config", **tc)

    log.info(f"device={device}")

    # data
    train_ds = PretrainingDataset(tc["train_bin"], mc["context_length"])
    val_ds = None
    if os.path.exists(tc["val_bin"]):
        val_ds = PretrainingDataset(tc["val_bin"], mc["context_length"])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # model
    model = ParrotLLM(model_config_dict).to(device)

    # ── Architecture log (slide 12 style) ─────────────────────────────────────
    _log_model_architecture(log, jlog, model, mc, device, tc["batch_size"])

    # torch.compile
    if device.type == "cuda":
        log.info("compiling model with torch.compile...")
        model = torch.compile(model)

    # optimizer
    optimizer = build_optimizer(model, tc)
    autocast_ctx, scaler = get_autocast_context(device)

    # resume
    start_step = 0
    if checkpoint:
        start_step, _ = load_checkpoint(checkpoint, model, optimizer, scaler, device)
        log.info(f"resumed from step {start_step}")

    # ── initial evaluation ────────────────────────────────────────────────────
    if val_ds is not None:
        log.info("Starting evaluation...")
        val_metrics = estimate_loss(model, val_ds, device, autocast_ctx, tc["batch_size"])
        jlog.log("pretraining", "initial_validation",
                 val_loss=val_metrics["loss"], val_ppl=val_metrics["perplexity"])
        log.info(f"  Initial val: loss={val_metrics['loss']:.4f}, ppl={val_metrics['perplexity']:.2f}")

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    grad_accum = tc["gradient_accumulation_steps"]
    steps_per_epoch = len(train_loader) // grad_accum if len(train_loader) > 0 else 1
    data_iter = iter(train_loader)
    train_start = time.time()
    t0 = train_start
    best_val_loss = float("inf")
    current_epoch = 0
    prev_epoch = 0
    loss_history: list[tuple[int, float]] = []
    
    log.info(fmt_training_start(steps_per_epoch, tc["max_steps"]))

    for step in range(start_step, tc["max_steps"]):
        epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0

        lr = get_lr(step, tc["warmup_steps"], tc["max_steps"],
                    tc["learning_rate"], tc["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                _, loss = model(x, targets=y)
                loss = loss / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        # gradient norm (for debug logging)
        grad_norm = 0.0
        if tc["grad_clip"] > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        ppl = math.exp(accum_loss) if accum_loss == accum_loss else float("nan")

        # console/file log (every log_every steps)
        if step % tc["log_every"] == 0:
            dt = time.time() - t0
            log.info(
                f"step {step:>6d} | epoch {epoch} | "
                f"loss {accum_loss:.4f} | lr {lr:.2e} | grad {grad_norm:.4f}"
            )
            log.debug(
                f"step {step:>6d} | ppl {ppl:.2f} | dt {dt:.1f}s"
            )
            t0 = time.time()

        # JSONL: every step (for plots)
        jlog.log("pretraining", "step",
                 epoch=epoch, step=step,
                 train_loss=accum_loss, perplexity=ppl, lr=lr)

        loss_history.append((step, accum_loss))

        # epoch boundary detection
        is_epoch_boundary = epoch > prev_epoch
        is_eval_step = (val_ds is not None and step > 0
                        and step % tc["eval_every"] == 0)

        if val_ds is not None and (is_epoch_boundary or is_eval_step):
            log.info("Starting evaluation...")
            log.info("-" * 60)
            val_metrics = estimate_loss(model, val_ds, device, autocast_ctx, tc["batch_size"])
            val_loss = val_metrics["loss"]
            val_ppl = val_metrics["perplexity"]

            if is_epoch_boundary:
                log.info(f"Epoch {prev_epoch + 1} complete:")
            log.info(f"  Train: loss={accum_loss:.4f}, ppl={ppl:.2f}")
            log.info(f"  Val:   loss={val_loss:.4f}, ppl={val_ppl:.2f}")

            jlog.log("pretraining", "eval",
                     step=step, epoch=epoch,
                     val_loss=val_loss, val_ppl=val_ppl)

            # Optuna intermediate reporting + pruning
            if trial is not None:
                trial.report(val_ppl, step)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log.info("  ** New best validation loss! **")
            log.info("-" * 60)

        if is_epoch_boundary:
            prev_epoch = epoch

        # checkpoint
        if step > 0 and step % tc["save_every"] == 0:
            save_checkpoint(model, optimizer, model_config_dict, step, epoch, scaler, run_dir)
            ckpt_path = os.path.join(run_dir, f"{epoch:02d}_epoch_{step}_step")
            log.info(f"Saved checkpoint: {ckpt_path}")
            jlog.log("pretraining", "checkpoint",
                     step=step, epoch=epoch, path=ckpt_path)

    # final save
    save_checkpoint(model, optimizer, model_config_dict, tc["max_steps"], epoch, scaler, run_dir)
    final_ckpt = os.path.join(run_dir, f"{epoch:02d}_epoch_{tc['max_steps']}_step")
    jlog.log("pretraining", "checkpoint",
             step=tc["max_steps"], epoch=epoch, path=final_ckpt)

    # ── ASCII loss curve ───────────────────────────────────────────────────────
    curve = render_ascii_loss_curve(loss_history)
    if curve:
        log.info("\n" + curve)

    # ── training complete summary ─────────────────────────────────────────────
    total_seconds = time.time() - train_start
    total_hours = total_seconds / 3600
    log.info(fmt_training_complete(
        epoch + 1, tc["max_steps"], total_hours, best_val_loss, run_dir,
    ))

    jlog.log("pretraining", "training_complete",
             epochs=epoch + 1,
             total_steps=tc["max_steps"],
             total_time_hours=round(total_hours, 2),
             best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
             run_dir=run_dir)

    jlog.close()
    log.info("done")

    best_val_ppl = math.exp(best_val_loss) if best_val_loss != float("inf") else float("inf")
    return best_val_ppl
