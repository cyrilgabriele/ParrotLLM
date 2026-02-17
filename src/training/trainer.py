"""Training loop for ParrotLLM pretraining."""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.model import ParrotLLM
from src.utils import get_device, load_config, set_seed


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
                    config: dict, step: int,
                    scaler: torch.amp.GradScaler | None,
                    checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
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
    print(f"[checkpoint] saved {path}")


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

def run_train(args) -> None:
    config = load_config(args.config)
    tc = config["training"]
    mc = config["model"]
    set_seed(tc["seed"])

    device = get_device(tc.get("device", args.device))
    print(f"[train] device={device}")

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
    model = ParrotLLM(config).to(device)
    print(f"[train] parameters: {model.count_parameters():,}")

    # torch.compile — ~20-40% speedup on CUDA (noop on MPS/CPU)
    if device.type == "cuda":
        print("[train] compiling model with torch.compile...")
        model = torch.compile(model)

    # optimizer
    optimizer = build_optimizer(model, tc)
    autocast_ctx, scaler = get_autocast_context(device)

    # resume
    start_step = 0
    if args.checkpoint:
        start_step, _ = load_checkpoint(args.checkpoint, model, optimizer, scaler, device)
        print(f"[train] resumed from step {start_step}")

    # wandb
    try:
        import wandb
        wandb.init(
            project=tc.get("wandb_project", "parrotllm"),
            name=tc.get("wandb_run_name"),
            config=config,
        )
        use_wandb = True
    except Exception:
        use_wandb = False
        print("[train] wandb not available, logging to console only")

    # training loop
    model.train()
    grad_accum = tc["gradient_accumulation_steps"]
    data_iter = iter(train_loader)
    t0 = time.time()

    for step in range(start_step, tc["max_steps"]):
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

        if tc["grad_clip"] > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # logging
        if step % tc["log_every"] == 0:
            dt = time.time() - t0
            print(f"step {step:>6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            if use_wandb:
                wandb.log({"train/loss": accum_loss, "train/lr": lr, "train/step": step})
            t0 = time.time()

        # eval
        if val_ds is not None and step > 0 and step % tc["eval_every"] == 0:
            val_metrics = estimate_loss(model, val_ds, device, autocast_ctx, tc["batch_size"])
            print(f"step {step:>6d} | val_loss {val_metrics['loss']:.4f} | val_ppl {val_metrics['perplexity']:.2f}")
            if use_wandb:
                wandb.log({
                    "val/loss": val_metrics["loss"],
                    "val/perplexity": val_metrics["perplexity"],
                    "train/step": step,
                })

        # checkpoint
        if step > 0 and step % tc["save_every"] == 0:
            save_checkpoint(model, optimizer, config, step, scaler, tc["checkpoint_dir"])

    # final save
    save_checkpoint(model, optimizer, config, tc["max_steps"], scaler, tc["checkpoint_dir"])
    if use_wandb:
        wandb.finish()
    print("[train] done")
