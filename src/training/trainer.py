"""Training loop for ParrotLLM pretraining."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from configs import ProjectConfig
from src.logging_utils import (
    JSONLLogger, TorchProfiler, fmt_model_summary, fmt_training_complete,
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
           max_lr: float, min_lr: float,
           schedule: str = "wsd", decay_ratio: float = 0.1) -> float:
    """Compute learning rate for the current step.

    Two schedules are supported:
    - "wsd" (Warmup-Stable-Decay, arXiv:2602.06797): linear warmup → constant
      plateau → linear decay to min_lr. The decay phase occupies `decay_ratio`
      of max_steps. More robust to changes in max_steps than cosine; linear
      decay-to-zero systematically outperforms cosine decay for LLMs.
    - "cosine": linear warmup → cosine annealing to min_lr. Legacy default.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    decay_steps = max(1, int(max_steps * decay_ratio))
    stable_end = max_steps - decay_steps

    if schedule == "wsd":
        if step < stable_end:
            return max_lr  # stable plateau
        progress = (step - stable_end) / decay_steps
        return max_lr + progress * (min_lr - max_lr)  # linear decay to min_lr
    else:  # cosine
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


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model (handles DDP + torch.compile wrappers)."""
    raw_model = model
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    return raw_model


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
    # unwrap compiled/DDP model if needed
    raw_model = _unwrap_model(model)
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
    raw_model = _unwrap_model(model)
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
                  max_batches: int = 20, *, num_workers: int = 0,
                  pin_memory: bool = False) -> dict:
    model.eval()
    losses = []
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
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


# ── Distributed helpers ─────────────────────────────────────────────────────

def _init_distributed(device: torch.device) -> tuple[torch.device, int, int, int, bool]:
    """Initialise torch.distributed if launched via torchrun."""
    if not dist.is_available():
        return device, 0, 1, 0, False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return device, 0, 1, 0, False

    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl":
            has_nccl = getattr(dist, "is_nccl_available", lambda: True)()
            if not has_nccl:
                logging.getLogger("parrotllm.training").warning(
                    "NCCL backend unavailable; falling back to gloo."
                )
                backend = "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if device.type == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    return device, rank, dist.get_world_size(), local_rank, True


def _broadcast_value(value, src: int = 0):
    """Broadcast a picklable value from src to all other ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


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
    pos_emb_params = 0  # RoPE has no learned positional embedding parameters
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

    device, rank, world_size, local_rank, distributed = _init_distributed(device)
    is_master = rank == 0
    trial_for_rank = trial if (trial is not None and (not distributed or is_master)) else None

    # ── run directory & loggers ───────────────────────────────────────────────
    run_dir = make_run_dir(tc_model.runs_dir) if is_master else None
    if distributed:
        run_dir = _broadcast_value(run_dir, src=0)
        dist.barrier()
    if is_master:
        setup_logger(
            run_dir,
            console_level=lc_model.console_level,
            file_level=lc_model.file_level,
            component_levels=lc_model.components if lc_model.components else None,
        )
    log = logging.getLogger("parrotllm.training")
    jlog: JSONLLogger | None = JSONLLogger(run_dir) if is_master else None

    profiler_cfg = None
    profiler_enabled_rank = False
    if lc_model and lc_model.profiler:
        profiler_cfg = lc_model.profiler.model_dump(mode="python")
        profiler_enabled_rank = (
            profiler_cfg.get("run_on_all_ranks", False) or is_master
        )
    profiler = TorchProfiler(
        config=profiler_cfg,
        run_dir=run_dir,
        logger=log,
        json_logger=jlog if is_master else None,
        enabled=profiler_enabled_rank,
    )

    # Save full config to run directory for reproducibility
    if is_master:
        config_path = os.path.join(run_dir, "config.json")
        json_payload = project_config.model_dump(mode="json")
        with open(config_path, "w") as f:
            json.dump(json_payload, f, indent=2)
        jlog.log("pretraining", "config", **tc)

    log_prefix = (
        f"device={device} | rank={rank} | world_size={world_size} | "
        f"distributed={'yes' if distributed else 'no'}"
    )
    if is_master:
        log.info(log_prefix)

    # data
    train_ds = PretrainingDataset(tc["train_bin"], mc["context_length"])
    val_ds = None
    if os.path.exists(tc["val_bin"]) and (not distributed or is_master):
        val_ds = PretrainingDataset(tc["val_bin"], mc["context_length"])

    pin_memory = bool(tc.get("pin_memory", True)) and device.type == "cuda"
    num_workers = int(tc.get("num_workers", 0))
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=distributed,
        persistent_workers=num_workers > 0,
    )

    # model
    model = ParrotLLM(model_config_dict).to(device)

    # ── Architecture log (slide 12 style) ─────────────────────────────────────
    if is_master and jlog is not None:
        _log_model_architecture(log, jlog, model, mc, device, tc["batch_size"])

    # torch.compile (skip if compile=false in config — useful for short HP tuning trials)
    use_compile = tc.get("compile", True)
    if device.type == "cuda" and use_compile:
        if is_master:
            log.info("compiling model with torch.compile...")
        model = torch.compile(model)
    elif device.type == "cuda" and not use_compile:
        if is_master:
            log.info("torch.compile disabled via config")

    if distributed:
        ddp_kwargs: dict = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
        model = DDP(model, **ddp_kwargs)

    # optimizer
    optimizer = build_optimizer(model, tc)
    autocast_ctx, scaler = get_autocast_context(device)

    # resume
    start_step = 0
    if checkpoint:
        start_step, _ = load_checkpoint(checkpoint, model, optimizer, scaler, device)
        if is_master:
            log.info(f"resumed from step {start_step}")

    # ── initial evaluation ────────────────────────────────────────────────────
    if val_ds is not None:
        log.info("Starting evaluation...")
        val_metrics = estimate_loss(
            model, val_ds, device, autocast_ctx, tc["batch_size"],
            num_workers=num_workers, pin_memory=pin_memory,
        )
        if jlog is not None:
            jlog.log(
                "pretraining", "initial_validation",
                val_loss=val_metrics["loss"], val_ppl=val_metrics["perplexity"],
            )
        log.info(
            "  Initial val: loss=%.4f, ppl=%.2f",
            val_metrics['loss'], val_metrics['perplexity'],
        )

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    grad_accum = tc["gradient_accumulation_steps"]
    total_micro_batches = len(train_loader)
    steps_per_epoch = max(1, total_micro_batches // max(1, grad_accum))
    data_iter = iter(train_loader)
    train_start = time.time()
    t0 = train_start
    best_val_loss = float("inf")
    initial_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    prev_epoch = initial_epoch
    loss_history: list[tuple[int, float]] = []
    if train_sampler is not None:
        train_sampler.set_epoch(initial_epoch)

    if is_master:
        log.info(fmt_training_start(steps_per_epoch, tc["max_steps"]))
        log.info(
            f"LR schedule={tc.get('lr_schedule', 'wsd')} | "
            f"decay_ratio={tc.get('lr_decay_ratio', 0.1)} | "
            f"z_loss_coeff={tc.get('z_loss_coeff', 0.0)}"
        )

    with profiler:
        for step in range(start_step, tc["max_steps"]):
            epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
            if train_sampler is not None and epoch > prev_epoch:
                train_sampler.set_epoch(epoch)

            with profiler.record_function("train.step"):
                lr = get_lr(
                    step, tc["warmup_steps"], tc["max_steps"],
                    tc["learning_rate"], tc["min_lr"],
                    schedule=tc.get("lr_schedule", "wsd"),
                    decay_ratio=tc.get("lr_decay_ratio", 0.1),
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                z_loss_coeff = tc.get("z_loss_coeff", 0.0)

                for micro in range(grad_accum):
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        x, y = next(data_iter)

                    x, y = x.to(device), y.to(device)
                    with autocast_ctx:
                        logits, ce_loss = model(x, targets=y)
                        if z_loss_coeff > 0.0:
                            # Z-loss (arXiv:2202.08906): penalises large pre-softmax logits to
                            # prevent numerical instability in mixed precision. Operates in
                            # float32 to avoid precision issues, then cast back.
                            z_loss = z_loss_coeff * torch.logsumexp(
                                logits.float(), dim=-1
                            ).pow(2).mean()
                            loss = (ce_loss + z_loss) / grad_accum
                        else:
                            loss = ce_loss / grad_accum

                    sync_grad = (not distributed) or (micro == grad_accum - 1)
                    ctx = (
                        model.no_sync
                        if (distributed and hasattr(model, "no_sync") and not sync_grad)
                        else nullcontext
                    )
                    with ctx():
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
            if is_master and step % tc["log_every"] == 0:
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
            if jlog is not None:
                jlog.log(
                    "pretraining", "step",
                    epoch=epoch, step=step,
                    train_loss=accum_loss, perplexity=ppl, lr=lr,
                )

            if is_master:
                loss_history.append((step, accum_loss))

            profiler.step(step=step, epoch=epoch)

            # epoch boundary detection
            is_epoch_boundary = epoch > prev_epoch
            is_eval_step = (val_ds is not None and step > 0
                            and step % tc["eval_every"] == 0)
            prune_this_step = False

            if val_ds is not None and (is_epoch_boundary or is_eval_step):
                log.info("Starting evaluation...")
                log.info("-" * 60)
                val_metrics = estimate_loss(
                    model, val_ds, device, autocast_ctx, tc["batch_size"],
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                val_loss = val_metrics["loss"]
                val_ppl = val_metrics["perplexity"]

                if is_epoch_boundary:
                    log.info(f"Epoch {prev_epoch + 1} complete:")
                log.info(f"  Train: loss={accum_loss:.4f}, ppl={ppl:.2f}")
                log.info(f"  Val:   loss={val_loss:.4f}, ppl={val_ppl:.2f}")

                if jlog is not None:
                    jlog.log(
                        "pretraining", "eval",
                        step=step, epoch=epoch,
                        val_loss=val_loss, val_ppl=val_ppl,
                    )

                if trial_for_rank is not None:
                    trial_for_rank.report(val_ppl, step)
                    if trial_for_rank.should_prune():
                        prune_this_step = True

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    log.info("  ** New best validation loss! **")
                log.info("-" * 60)

            if distributed and trial is not None:
                flag = torch.tensor(1 if (is_master and prune_this_step) else 0, device=device)
                dist.broadcast(flag, src=0)
                if not is_master and flag.item() == 1:
                    import optuna
                    raise optuna.TrialPruned()

            if prune_this_step:
                import optuna
                raise optuna.TrialPruned()

            if is_epoch_boundary:
                prev_epoch = epoch

            # checkpoint
            if is_master and step > 0 and step % tc["save_every"] == 0:
                save_checkpoint(model, optimizer, model_config_dict, step, epoch, scaler, run_dir)
                ckpt_path = os.path.join(run_dir, f"{epoch:02d}_epoch_{step}_step")
                log.info(f"Saved checkpoint: {ckpt_path}")
                if jlog is not None:
                    jlog.log(
                        "pretraining", "checkpoint",
                        step=step, epoch=epoch, path=ckpt_path,
                    )

    # final save
    if is_master:
        save_checkpoint(model, optimizer, model_config_dict, tc["max_steps"], epoch, scaler, run_dir)
        final_ckpt = os.path.join(run_dir, f"{epoch:02d}_epoch_{tc['max_steps']}_step")
        if jlog is not None:
            jlog.log(
                "pretraining", "checkpoint",
                step=tc["max_steps"], epoch=epoch, path=final_ckpt,
            )

    # ── ASCII loss curve ───────────────────────────────────────────────────────
    if is_master:
        curve = render_ascii_loss_curve(loss_history)
        if curve:
            log.info("\n" + curve)

    # ── training complete summary ─────────────────────────────────────────────
    total_seconds = time.time() - train_start
    total_hours = total_seconds / 3600
    if is_master:
        log.info(fmt_training_complete(
            epoch + 1, tc["max_steps"], total_hours, best_val_loss, run_dir,
        ))

    if jlog is not None:
        jlog.log(
            "pretraining", "training_complete",
            epochs=epoch + 1,
            total_steps=tc["max_steps"],
            total_time_hours=round(total_hours, 2),
            best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
            run_dir=run_dir,
        )
        jlog.close()

    if is_master:
        log.info("done")

    if distributed:
        tensor = torch.tensor(best_val_loss, device=device)
        dist.broadcast(tensor, src=0)
        best_val_loss = tensor.item()

    best_val_ppl = math.exp(best_val_loss) if best_val_loss != float("inf") else float("inf")
    return best_val_ppl
