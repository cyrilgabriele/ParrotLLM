"""Training loop for ParrotLLM pretraining."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import optuna

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler
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

def _resolve_wsd_decay_steps(
    *,
    warmup_steps: int,
    max_steps: int,
    decay_ratio: float,
) -> tuple[int, int]:
    """Return the WSD plateau end step and decay length.

    The decay phase is clamped to the post-warmup portion of training so the
    warmup and decay phases never overlap.
    """
    post_warmup_steps = max(0, max_steps - warmup_steps)
    requested_decay_steps = int(max_steps * decay_ratio)
    decay_steps = min(post_warmup_steps, max(0, requested_decay_steps))
    stable_end = max_steps - decay_steps
    return stable_end, decay_steps


def compute_lr(step: int, warmup_steps: int, max_steps: int,
               max_lr: float, min_lr: float,
               schedule: str = "wsd", decay_ratio: float = 0.1) -> float:
    """Compute the learning rate for a zero-based optimizer step.

    The final real training step (`step == max_steps - 1`) reaches `min_lr`.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if max_lr <= 0.0:
        raise ValueError("max_lr must be positive.")
    if min_lr < 0.0:
        raise ValueError("min_lr must be non-negative.")
    if min_lr > max_lr:
        raise ValueError("min_lr must not exceed max_lr.")
    if not 0.0 <= decay_ratio <= 1.0:
        raise ValueError("decay_ratio must be between 0 and 1.")
    if schedule not in {"wsd", "cosine"}:
        raise ValueError(f"Unsupported lr schedule: {schedule}")

    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr

    if schedule == "wsd":
        stable_end, decay_steps = _resolve_wsd_decay_steps(
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            decay_ratio=decay_ratio,
        )
        if decay_steps == 0 or step < stable_end:
            return max_lr
        progress = (step - stable_end + 1) / decay_steps
        return max_lr + progress * (min_lr - max_lr)

    cosine_steps = max_steps - min(max_steps, warmup_steps)
    if cosine_steps <= 0:
        return max_lr

    progress = (step - warmup_steps + 1) / cosine_steps
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class ParrotLRScheduler(LRScheduler):
    """PyTorch scheduler wrapper for ParrotLLM's warmup + WSD/cosine schedules."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
        schedule: str = "wsd",
        decay_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.schedule = schedule
        self.decay_ratio = float(decay_ratio)
        self.min_lrs = [float(min_lr)] * len(optimizer.param_groups)

        if self.max_steps <= 0:
            raise ValueError("training.max_steps must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative.")
        if self.schedule not in {"wsd", "cosine"}:
            raise ValueError(f"Unsupported training.lr_schedule: {self.schedule}")
        if not 0.0 <= self.decay_ratio <= 1.0:
            raise ValueError("training.lr_decay_ratio must be between 0 and 1.")

        for idx, (group, group_min_lr) in enumerate(zip(optimizer.param_groups, self.min_lrs)):
            group_lr = float(group["lr"])
            if group_lr <= 0.0:
                raise ValueError(f"Optimizer lr for param group {idx} must be positive.")
            if group_min_lr > group_lr:
                raise ValueError(
                    f"training.min_lr ({group_min_lr}) must not exceed optimizer lr ({group_lr}) "
                    f"for param group {idx}."
                )

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lrs_for_step(self, step: int) -> list[float]:
        return [
            compute_lr(
                step,
                self.warmup_steps,
                self.max_steps,
                base_lr,
                min_lr,
                schedule=self.schedule,
                decay_ratio=self.decay_ratio,
            )
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

    def get_lr(self) -> list[float]:
        return self._lrs_for_step(self.last_epoch)

    def _get_closed_form_lr(self) -> list[float]:
        return self._lrs_for_step(self.last_epoch)

    def resume_to_step(self, step: int) -> None:
        """Fast-forward the scheduler to the next optimizer step after resume."""
        target_step = max(0, int(step))
        lrs = self._lrs_for_step(target_step)
        self.last_epoch = target_step
        self._step_count = target_step + 1
        self._last_lr = lrs
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr


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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    tc: dict,
) -> ParrotLRScheduler:
    return ParrotLRScheduler(
        optimizer,
        warmup_steps=tc["warmup_steps"],
        max_steps=tc["max_steps"],
        min_lr=tc["min_lr"],
        schedule=tc.get("lr_schedule", "wsd"),
        decay_ratio=tc.get("lr_decay_ratio", 0.1),
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
                    checkpoint_dir: str,
                    filename: str | None = None,
                    scheduler: LRScheduler | None = None,
                    trainer_state: dict | None = None) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = (
        os.path.join(checkpoint_dir, filename)
        if filename is not None
        else os.path.join(checkpoint_dir, f"{epoch:02d}_epoch_{step}_step")
    )
    # unwrap compiled/DDP model if needed
    raw_model = _unwrap_model(model)
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if trainer_state is not None:
        state["trainer_state"] = trainer_state
    torch.save(state, path)
    logging.getLogger("parrotllm.training").debug(f"Checkpoint saved: {path}")
    return path


def resolve_checkpoint_dir(run_dir: str, checkpoint_dir: str) -> str:
    """Resolve a checkpoint subdirectory under the current run directory."""
    if os.path.isabs(checkpoint_dir):
        raise ValueError(
            "training.checkpoint_dir must be a relative path inside the run directory."
        )

    resolved = os.path.abspath(os.path.join(run_dir, checkpoint_dir))
    run_root = os.path.abspath(run_dir)
    if os.path.commonpath([run_root, resolved]) != run_root:
        raise ValueError(
            "training.checkpoint_dir must stay inside the run directory."
        )

    os.makedirs(resolved, exist_ok=True)
    return resolved


@dataclass
class CheckpointRecord:
    path: str
    step: int
    epoch: int
    val_loss: float | None = None


class CheckpointManager:
    """Save run-local checkpoints and retain only the configured snapshots."""

    def __init__(self, checkpoint_dir: str, *, keep_last: int = 10, keep_best: int = 10):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = max(0, int(keep_last))
        self.keep_best = max(0, int(keep_best))
        self._last_records: list[CheckpointRecord] = []
        self._best_records: list[CheckpointRecord] = []

    @property
    def best_path(self) -> str | None:
        if not self._best_records:
            return None
        return self._best_records[0].path

    def save_last(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        step: int,
        epoch: int,
        scaler: torch.amp.GradScaler | None,
        scheduler: LRScheduler | None = None,
        trainer_state: dict | None = None,
    ) -> str | None:
        if self.keep_last <= 0:
            return None

        filename = f"last_epoch_{epoch:04d}_step_{step:07d}.pt"
        path = save_checkpoint(
            model,
            optimizer,
            config,
            step,
            epoch,
            scaler,
            self.checkpoint_dir,
            filename=filename,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        record = CheckpointRecord(path=path, step=step, epoch=epoch)
        self._last_records = [r for r in self._last_records if r.path != path]
        self._last_records.append(record)
        self._last_records.sort(key=lambda r: (r.step, r.epoch, r.path))
        self._prune_last()
        return path

    def maybe_save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        step: int,
        epoch: int,
        scaler: torch.amp.GradScaler | None,
        val_loss: float,
        scheduler: LRScheduler | None = None,
        trainer_state: dict | None = None,
    ) -> str | None:
        if self.keep_best <= 0 or not math.isfinite(val_loss):
            return None

        if len(self._best_records) >= self.keep_best:
            worst = max(
                self._best_records,
                key=lambda r: (
                    r.val_loss if r.val_loss is not None else float("inf"),
                    r.step,
                    r.path,
                ),
            )
            if worst.val_loss is not None and val_loss >= worst.val_loss:
                return None

        filename = (
            f"best_loss_{self._format_loss(val_loss)}_"
            f"epoch_{epoch:04d}_step_{step:07d}.pt"
        )
        path = save_checkpoint(
            model,
            optimizer,
            config,
            step,
            epoch,
            scaler,
            self.checkpoint_dir,
            filename=filename,
            scheduler=scheduler,
            trainer_state=trainer_state,
        )
        record = CheckpointRecord(path=path, step=step, epoch=epoch, val_loss=val_loss)
        self._best_records = [r for r in self._best_records if r.path != path]
        self._best_records.append(record)
        self._best_records.sort(
            key=lambda r: (
                r.val_loss if r.val_loss is not None else float("inf"),
                r.step,
                r.path,
            )
        )
        self._prune_best()
        return path

    def _prune_last(self) -> None:
        while len(self._last_records) > self.keep_last:
            stale = self._last_records.pop(0)
            self._remove_file(stale.path)

    def _prune_best(self) -> None:
        while len(self._best_records) > self.keep_best:
            stale = self._best_records.pop(-1)
            self._remove_file(stale.path)

    @staticmethod
    def _remove_file(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _format_loss(val_loss: float) -> str:
        return f"{val_loss:.4f}".replace(".", "p")


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scaler: torch.amp.GradScaler | None = None,
                    device: torch.device = torch.device("cpu"),
                    scheduler: LRScheduler | None = None,
                    *,
                    return_trainer_state: bool = False):
    # Always deserialise to CPU first. load_state_dict() then copies each
    # tensor to the correct device (model parameters and optimiser states live
    # on `device`). Loading directly to an accelerator device (MPS/CUDA) would
    # place the *entire* raw checkpoint — model weights + all AdamW moment
    # buffers — on the device simultaneously, which can spike memory by 3–4×
    # the model size and cause OOM before training even starts.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = _unwrap_model(model)
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("step", 0))
    if scheduler is not None:
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        elif hasattr(scheduler, "resume_to_step"):
            logging.getLogger("parrotllm.training").warning(
                "Checkpoint %s has no scheduler state; reconstructing the LR position from "
                "the saved step. Legacy periodic checkpoints saved before this change may "
                "resume one step early.",
                path,
            )
            scheduler.resume_to_step(step)
    if return_trainer_state:
        return step, ckpt.get("config", {}), ckpt.get("trainer_state")
    return step, ckpt.get("config", {})


# ── Checkpoint discovery ─────────────────────────────────────────────────────

# Matches new format:  last_epoch_0001_step_0015000.pt  → group 1
# Matches old format:  00_epoch_1000_step               → group 2
_STEP_RE = re.compile(r"_step_(\d+)\.pt$|_epoch_(\d+)_step(?:\.|$)")

# Old-format checkpoint: no extension, e.g. 00_epoch_1000_step
_OLD_CKPT_RE = re.compile(r"^\d+_epoch_\d+_step$")


def _parse_step_from_filename(filename: str) -> int | None:
    """Extract the optimizer step from a checkpoint filename, or None if unparseable.

    Handles both the current ``last_epoch_NNNN_step_NNNNNNN.pt`` format and
    the legacy ``NN_epoch_NNN_step`` format (no file extension).
    """
    m = _STEP_RE.search(filename)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def _is_checkpoint_candidate(filename: str) -> bool:
    """Return True for filenames that could be a checkpoint saved by save_checkpoint()."""
    return filename.endswith(".pt") or bool(_OLD_CKPT_RE.match(filename))


def _peek_checkpoint_model_config(path: str) -> dict | None:
    """Load a checkpoint and return its saved model-config dict, or None on failure.

    Used by :func:`run_train` to reconstruct the exact model architecture that
    was used when the checkpoint was saved, so that resuming does not require
    the user to keep the YAML config manually in sync with the checkpoint.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        if isinstance(cfg, dict) and "model" in cfg and isinstance(cfg["model"], dict):
            return cfg["model"]
        return None
    except Exception:
        return None


def _validate_checkpoint(path: str) -> tuple[bool, str]:
    """Check whether a checkpoint file is loadable and structurally valid.

    Loads the file on CPU to avoid occupying GPU memory during discovery.
    Returns ``(True, "")`` on success, or ``(False, reason)`` on failure so
    the caller can log a precise diagnostic instead of a silent skip.

    A valid checkpoint must be a ``dict`` with at least ``"model"`` and
    ``"step"`` keys — the minimum required to resume training.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            return False, f"payload is {type(ckpt).__name__!r}, expected dict"
        missing = [k for k in ("model", "step") if k not in ckpt]
        if missing:
            return False, f"missing required keys: {missing!r}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def find_latest_checkpoint(
    runs_dir: str,
    run_dir: str | None = None,
    *,
    checkpoint_subdir: str = "checkpoints",
    validate: bool = True,
) -> str:
    """Find the latest valid checkpoint to resume training from.

    Searches *checkpoint_subdir* inside the target run directory for
    ``*.pt`` files, picks the one with the highest step number, and —
    when *validate* is ``True`` — verifies it can be loaded before
    returning its path. Corrupted or truncated files are skipped
    automatically and the next-best candidate is tried.

    Run-directory selection
    -----------------------
    * If *run_dir* is given, that directory is used directly.
    * Otherwise the most recently created ``run_*`` directory inside
      *runs_dir* is selected (directories are sorted lexicographically
      so the ``run_YYYYMMDD_HHMMSS`` timestamp determines recency).

    Checkpoint ranking
    ------------------
    Candidates are ranked by:

    1. **Step number** (descending) — resume as close to the interruption
       as possible.
    2. **Checkpoint type** (``last_*`` beats ``best_*`` at equal step) —
       periodic saves capture full training state regardless of val-loss.
    3. **File mtime** (descending) — tie-break for non-standard names.

    Args:
        runs_dir: Base directory containing ``run_*`` sub-directories
            created by :func:`make_run_dir`.
        run_dir: Specific run directory to search. When ``None`` the most
            recent ``run_*`` directory inside *runs_dir* is used.
        checkpoint_subdir: Sub-directory name inside each run directory
            where checkpoints are stored (default: ``"checkpoints"``).
        validate: Verify each candidate is loadable. Corrupted files are
            skipped and the next candidate is tried.

    Returns:
        Absolute path to the latest valid checkpoint file.

    Raises:
        FileNotFoundError: When the directory structure is missing, no
            checkpoint files exist, or every candidate fails validation.
    """
    log = logging.getLogger("parrotllm.training")

    # ── 1. Build ordered list of run directories to search ───────────────────
    if run_dir is not None:
        # Explicit path: search only that directory, no fallback.
        run_dir = os.path.abspath(run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(
                f"Specified run directory does not exist: {run_dir!r}"
            )
        run_dirs_to_search = [run_dir]
    else:
        runs_root = os.path.abspath(runs_dir)
        if not os.path.isdir(runs_root):
            raise FileNotFoundError(
                f"Runs directory does not exist: {runs_root!r}. "
                "Start a fresh training run first, or pass an explicit "
                "--resume <run_dir> path."
            )
        run_candidates = sorted(
            (
                d for d in os.listdir(runs_root)
                if d.startswith("run_")
                and os.path.isdir(os.path.join(runs_root, d))
            ),
            reverse=True,  # lexicographic descending ≡ newest first
        )
        if not run_candidates:
            raise FileNotFoundError(
                f"No run_* directories found in {runs_root!r}. "
                "Start a fresh training run first."
            )
        run_dirs_to_search = [os.path.join(runs_root, d) for d in run_candidates]

    # ── 2. Search run directories in order until a valid checkpoint is found ──
    def _collect_candidates(search_run_dir: str) -> list[str]:
        """Return all checkpoint candidate file paths from *search_run_dir*.

        Checks two locations:
        - ``<run_dir>/<checkpoint_subdir>/`` — current layout (*.pt files)
        - ``<run_dir>/`` root — legacy layout (no extension, NN_epoch_NNN_step)
        """
        candidates: list[str] = []
        for search_path in (
            os.path.join(search_run_dir, checkpoint_subdir),
            search_run_dir,
        ):
            if not os.path.isdir(search_path):
                continue
            for f in os.listdir(search_path):
                if _is_checkpoint_candidate(f):
                    candidates.append(os.path.join(search_path, f))
        return candidates

    def _rank(path: str) -> tuple[int, int, float]:
        name = os.path.basename(path)
        step = _parse_step_from_filename(name)
        is_last = 1 if name.startswith("last_") else 0
        mtime = os.path.getmtime(path)
        return (step if step is not None else -1, is_last, mtime)

    def _find_in_run(search_run_dir: str) -> str | None:
        """Return the best valid checkpoint path from *search_run_dir*, or None."""
        candidates = _collect_candidates(search_run_dir)
        if not candidates:
            return None

        candidates.sort(key=_rank, reverse=True)

        for path in candidates:
            if validate:
                ok, reason = _validate_checkpoint(path)
                if not ok:
                    log.warning(
                        "Skipping checkpoint %s — validation failed: %s",
                        path,
                        reason,
                    )
                    continue
            return path

        return None

    skipped: list[str] = []
    for search_dir in run_dirs_to_search:
        path = _find_in_run(search_dir)
        if path is not None:
            if skipped:
                log.warning(
                    "%d run(s) skipped (no usable checkpoints): %s",
                    len(skipped),
                    ", ".join(os.path.basename(d) for d in skipped),
                )
            log.info("Auto-selected run directory for resume: %s", search_dir)
            step = _parse_step_from_filename(os.path.basename(path))
            log.info(
                "Selected checkpoint: %s (step=%s)",
                path,
                step if step is not None else "unknown",
            )
            return path
        skipped.append(search_dir)

    if run_dir is not None:
        # Explicit run_dir — give a targeted error showing what was found.
        candidates = _collect_candidates(run_dirs_to_search[0])
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint files found in {run_dirs_to_search[0]!r} "
                f"(checked '{checkpoint_subdir}/' subdir and run root). "
                "The run may not have saved any checkpoints yet."
            )
        raise FileNotFoundError(
            f"Found {len(candidates)} checkpoint file(s) in {run_dirs_to_search[0]!r} "
            "but none passed validation. Run with --resume without a path to search "
            "older runs, or check the files manually."
        )

    raise FileNotFoundError(
        f"No valid checkpoint found in any of the {len(skipped)} run(s) under "
        f"{os.path.abspath(runs_dir)!r}. "
        "All runs either have no checkpoint files or every file failed validation. "
        "Check the WARNING lines above for per-file failure reasons."
    )


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


def _empty_device_cache(device: torch.device) -> None:
    """Release any cached-but-unused memory back to the device allocator.

    Called after checkpoint loading to ensure transient CPU→device copies are
    freed before the initial evaluation allocates activations.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _checkpoint_trainer_state(
    *,
    next_epoch: int,
    next_micro_batch: int,
    data_seed: int,
) -> dict[str, int]:
    """Serialize the next dataloader position for exact resume."""
    return {
        "next_epoch": int(next_epoch),
        "next_micro_batch": int(next_micro_batch),
        "data_seed": int(data_seed),
    }


def _restore_data_position(
    trainer_state: dict | None,
    *,
    start_step: int,
    grad_accum: int,
    total_micro_batches: int,
) -> tuple[int, int, int, bool]:
    """Return the next epoch/batch position and whether it is exact.

    New checkpoints store the exact dataloader position. Legacy checkpoints do
    not, so we approximate from completed optimizer steps and grad accumulation.
    """
    data_seed = int(torch.initial_seed())
    if trainer_state is not None:
        data_seed = int(trainer_state.get("data_seed", data_seed))
        next_epoch = int(trainer_state.get("next_epoch", 0))
        next_micro_batch = int(trainer_state.get("next_micro_batch", 0))
        if total_micro_batches > 0:
            next_epoch += next_micro_batch // total_micro_batches
            next_micro_batch %= total_micro_batches
        return next_epoch, next_micro_batch, data_seed, True

    if total_micro_batches <= 0:
        return 0, 0, data_seed, False

    consumed_micro_batches = max(0, int(start_step)) * max(1, int(grad_accum))
    next_epoch = consumed_micro_batches // total_micro_batches
    next_micro_batch = consumed_micro_batches % total_micro_batches
    return next_epoch, next_micro_batch, data_seed, False


def _build_epoch_iterator(
    train_loader: torch.utils.data.DataLoader,
    *,
    epoch: int,
    train_sampler: DistributedSampler | None,
    shuffle_generator: torch.Generator | None,
    data_seed: int,
    skip_micro_batches: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Create a deterministic iterator for one training epoch."""
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    elif shuffle_generator is not None:
        shuffle_generator.manual_seed(data_seed + epoch)

    data_iter = iter(train_loader)
    for _ in range(skip_micro_batches):
        try:
            next(data_iter)
        except StopIteration as exc:  # pragma: no cover - config/state mismatch
            raise ValueError(
                "Checkpoint resume position exceeds the available training batches."
            ) from exc
    return data_iter


def _apply_optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> bool:
    """Run the optimizer step and report whether parameters were updated."""
    if scaler is None:
        optimizer.step()
        return True

    previous_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    current_scale = float(scaler.get_scale())
    return current_scale >= previous_scale


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

    # ── Resolve model architecture from checkpoint (must happen before model build) ──
    # When resuming, the checkpoint's saved model config is the source of truth for
    # architecture. Using the current YAML config would cause a shape mismatch if the
    # architecture changed between runs.
    if checkpoint is not None:
        ckpt_mc = _peek_checkpoint_model_config(checkpoint)
        if ckpt_mc is not None:
            arch_keys = ("d_model", "n_layers", "n_heads", "d_ff", "context_length",
                         "vocab_size", "bias", "rope_theta")
            diffs = {
                k: (mc.get(k), ckpt_mc.get(k))
                for k in arch_keys
                if mc.get(k) != ckpt_mc.get(k)
            }
            if diffs:
                diff_lines = "\n".join(
                    f"  {k}: config={yaml_val!r} → checkpoint={ckpt_val!r}"
                    for k, (yaml_val, ckpt_val) in diffs.items()
                )
                # We don't have a logger yet (setup_logger runs later), use root logger.
                logging.warning(
                    "Checkpoint architecture differs from current config — "
                    "using checkpoint values to build the model:\n%s",
                    diff_lines,
                )
            mc = {**mc, **ckpt_mc}
            model_config_dict = {**model_config_dict, "model": mc}

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
    checkpoint_dir = resolve_checkpoint_dir(run_dir, tc["checkpoint_dir"])
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        keep_last=tc.get("keep_last_checkpoints", 10),
        keep_best=tc.get("keep_best_checkpoints", 10),
    )

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
        jlog.log(
            "pretraining",
            "checkpoint_policy",
            checkpoint_dir=checkpoint_dir,
            keep_last=checkpoint_manager.keep_last,
            keep_best=checkpoint_manager.keep_best,
        )

    log_prefix = (
        f"device={device} | rank={rank} | world_size={world_size} | "
        f"distributed={'yes' if distributed else 'no'}"
    )
    if is_master:
        log.info(log_prefix)
        log.info(
            "checkpoint directory=%s | keep_last=%d | keep_best=%d",
            checkpoint_dir,
            checkpoint_manager.keep_last,
            checkpoint_manager.keep_best,
        )

    # data
    train_ds = PretrainingDataset(tc["train_bin"], mc["context_length"])
    val_ds = None
    if os.path.exists(tc["val_bin"]) and (not distributed or is_master):
        val_ds = PretrainingDataset(tc["val_bin"], mc["context_length"])

    pin_memory = bool(tc.get("pin_memory", True)) and device.type == "cuda"
    num_workers = int(tc.get("num_workers", 0))
    train_sampler = None
    train_shuffle_generator = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
    else:
        train_shuffle_generator = torch.Generator()

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None,
        generator=train_shuffle_generator,
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
    scheduler = build_scheduler(optimizer, tc)
    autocast_ctx, scaler = get_autocast_context(device)

    # resume
    start_step = 0
    resume_epoch = 0
    resume_micro_batch = 0
    data_seed = int(torch.initial_seed())
    if checkpoint:
        start_step, _, trainer_state = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scaler,
            device,
            scheduler=scheduler,
            return_trainer_state=True,
        )
        resume_epoch, resume_micro_batch, data_seed, exact_data_resume = _restore_data_position(
            trainer_state,
            start_step=start_step,
            grad_accum=tc["gradient_accumulation_steps"],
            total_micro_batches=len(train_loader),
        )
        if is_master:
            log.info(
                "resumed from step %d | next_epoch=%d | next_micro_batch=%d",
                start_step,
                resume_epoch,
                resume_micro_batch,
            )
            if not exact_data_resume:
                log.warning(
                    "Checkpoint %s has no saved dataloader position; approximating resume "
                    "from step=%d and gradient_accumulation_steps=%d. Exact mid-epoch "
                    "continuation is available only for checkpoints saved after this change.",
                    checkpoint,
                    start_step,
                    tc["gradient_accumulation_steps"],
                )
            if start_step >= tc["max_steps"]:
                log.warning(
                    "Checkpoint step %d >= max_steps %d — training is already complete. "
                    "No additional steps will be taken. "
                    "Increase max_steps in the config to extend training.",
                    start_step,
                    tc["max_steps"],
                )

    # Release any unreferenced tensors so the allocator has headroom before the
    # first estimate_loss call (especially important after checkpoint loading,
    # where CPU→device copies of model and optimiser states may still occupy
    # cached memory).
    if checkpoint:
        import gc
        gc.collect()
        _empty_device_cache(device)

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
    tokens_per_step: int = tc["batch_size"] * mc["context_length"] * grad_accum
    total_micro_batches = len(train_loader)
    if total_micro_batches <= 0:
        raise ValueError("Training dataset produced zero batches.")
    steps_per_epoch = max(1, math.ceil(total_micro_batches / max(1, grad_accum)))
    train_start = time.time()
    t0 = train_start
    _step_t = train_start
    early_stopping_patience = int(tc.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    early_stopping_enabled = val_ds is not None and early_stopping_patience > 0
    best_val_loss = float("inf")
    best_checkpoint_path: str | None = None
    evals_without_improvement = 0
    completed_steps = start_step
    current_epoch = resume_epoch
    next_epoch = resume_epoch
    next_micro_batch = resume_micro_batch
    epoch_micro_batches_consumed = resume_micro_batch
    early_stop_triggered = False
    early_stop_reason: str | None = None
    loss_history: list[tuple[int, float]] = []
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None
    if completed_steps < tc["max_steps"]:
        data_iter = _build_epoch_iterator(
            train_loader,
            epoch=current_epoch,
            train_sampler=train_sampler,
            shuffle_generator=train_shuffle_generator,
            data_seed=data_seed,
            skip_micro_batches=resume_micro_batch,
        )

    if is_master:
        log.info(fmt_training_start(steps_per_epoch, tc["max_steps"]))
        log.info(
            f"LR schedule={tc.get('lr_schedule', 'wsd')} | "
            f"decay_ratio={tc.get('lr_decay_ratio', 0.1)} | "
            f"z_loss_coeff={tc.get('z_loss_coeff', 0.0)}"
        )
        if early_stopping_enabled:
            log.info(
                "Early stopping enabled | patience=%d evals | min_delta=%.6f",
                early_stopping_patience,
                early_stopping_min_delta,
            )

    with profiler:
        while completed_steps < tc["max_steps"]:
            if epoch_micro_batches_consumed >= total_micro_batches:
                current_epoch += 1
                epoch_micro_batches_consumed = 0
                data_iter = _build_epoch_iterator(
                    train_loader,
                    epoch=current_epoch,
                    train_sampler=train_sampler,
                    shuffle_generator=train_shuffle_generator,
                    data_seed=data_seed,
                )

            remaining_micro_batches = total_micro_batches - epoch_micro_batches_consumed
            micro_batches_target = min(grad_accum, remaining_micro_batches)
            with profiler.record_function("train.step"):
                lr = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                z_loss_coeff = tc.get("z_loss_coeff", 0.0)

                for micro in range(micro_batches_target):
                    if data_iter is None:  # pragma: no cover - guarded above
                        raise RuntimeError("Training iterator was not initialised.")
                    try:
                        x, y = next(data_iter)
                    except StopIteration as exc:  # pragma: no cover - iterator bookkeeping bug
                        raise RuntimeError(
                            "Training iterator exhausted before the tracked epoch position."
                        ) from exc

                    epoch_micro_batches_consumed += 1

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
                            loss = (ce_loss + z_loss) / micro_batches_target
                        else:
                            loss = ce_loss / micro_batches_target

                    sync_grad = (not distributed) or (micro == micro_batches_target - 1)
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

                optimizer_updated = _apply_optimizer_step(optimizer, scaler)
                _step_dt = time.time() - _step_t
                _step_t = time.time()
                if optimizer_updated:
                    scheduler.step()
                    completed_steps += 1

            ppl = math.exp(accum_loss) if accum_loss == accum_loss else float("nan")
            is_epoch_boundary = epoch_micro_batches_consumed == total_micro_batches
            next_epoch = current_epoch + 1 if is_epoch_boundary else current_epoch
            next_micro_batch = 0 if is_epoch_boundary else epoch_micro_batches_consumed
            trainer_state = _checkpoint_trainer_state(
                next_epoch=next_epoch,
                next_micro_batch=next_micro_batch,
                data_seed=data_seed,
            )

            if not optimizer_updated:
                if is_master:
                    log.warning(
                        "Optimizer step skipped at epoch %d after %d/%d micro-batches; "
                        "LR schedule and completed_steps remain unchanged.",
                        current_epoch,
                        epoch_micro_batches_consumed,
                        total_micro_batches,
                    )

            if optimizer_updated:
                should_log_step = (
                    completed_steps == start_step + 1
                    or completed_steps % tc["log_every"] == 0
                )

                if is_master and should_log_step:
                    dt = time.time() - t0
                    log.info(
                        f"step {completed_steps:>6d} | epoch {current_epoch} | "
                        f"loss {accum_loss:.4f} | lr {lr:.2e} | grad {grad_norm:.4f}"
                    )
                    log.debug(
                        f"step {completed_steps:>6d} | ppl {ppl:.2f} | dt {dt:.1f}s"
                    )
                    t0 = time.time()

                if jlog is not None:
                    jlog.log(
                        "pretraining", "step",
                        epoch=current_epoch, step=completed_steps,
                        train_loss=accum_loss, perplexity=ppl, lr=lr,
                        grad_norm=round(grad_norm, 6),
                        tokens_per_sec=round(tokens_per_step / _step_dt) if _step_dt > 0 else 0,
                    )

                if is_master:
                    loss_history.append((completed_steps, accum_loss))

                profiler.step(step=completed_steps, epoch=current_epoch)

            is_eval_step = (
                optimizer_updated
                and val_ds is not None
                and completed_steps > 0
                and completed_steps % tc["eval_every"] == 0
            )
            eval_epoch = next_epoch if is_epoch_boundary else current_epoch
            prune_this_step = False
            stop_training = False

            if val_ds is not None and (is_epoch_boundary or is_eval_step):
                log.info("Starting evaluation...")
                log.info("-" * 60)
                val_metrics = estimate_loss(
                    model, val_ds, device, autocast_ctx, tc["batch_size"],
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                val_loss = val_metrics["loss"]
                val_ppl = val_metrics["perplexity"]
                improved = False

                if is_epoch_boundary:
                    log.info(f"Epoch {eval_epoch} complete:")
                log.info(f"  Train: loss={accum_loss:.4f}, ppl={ppl:.2f}")
                log.info(f"  Val:   loss={val_loss:.4f}, ppl={val_ppl:.2f}")

                if jlog is not None:
                    jlog.log(
                        "pretraining", "eval",
                        step=completed_steps, epoch=eval_epoch,
                        val_loss=val_loss, val_ppl=val_ppl,
                        eval_train_loss=accum_loss, eval_train_ppl=ppl,
                    )

                if trial_for_rank is not None:
                    trial_for_rank.report(val_ppl, completed_steps)
                    if trial_for_rank.should_prune():
                        prune_this_step = True

                if not math.isfinite(val_loss):
                    stop_training = True
                    early_stop_reason = "validation loss became non-finite"
                    log.warning("  Validation loss is non-finite; stopping training.")
                elif val_loss < (best_val_loss - early_stopping_min_delta):
                    improved = True
                    best_val_loss = val_loss
                    evals_without_improvement = 0
                    log.info("  ** New best validation loss! **")
                if is_master:
                    best_candidate_path = checkpoint_manager.maybe_save_best(
                        model,
                        optimizer,
                        model_config_dict,
                        completed_steps,
                        eval_epoch,
                        scaler,
                        val_loss,
                        scheduler=scheduler,
                        trainer_state=trainer_state,
                    )
                    if best_candidate_path is not None:
                        log.info(f"  Saved best checkpoint: {best_candidate_path}")
                        if jlog is not None:
                            jlog.log(
                                "pretraining",
                                "best_checkpoint",
                                step=completed_steps,
                                epoch=eval_epoch,
                                path=best_candidate_path,
                                val_loss=val_loss,
                            )
                        try:
                            from pathlib import Path as _Path
                            from src.scripts.plot_training import plot_run_dir as _plot_run_dir
                            _plot_run_dir(_Path(run_dir))
                        except Exception as _exc:
                            log.debug("Auto-plotting skipped: %s", _exc)
                    best_checkpoint_path = checkpoint_manager.best_path
                if math.isfinite(val_loss) and (not improved) and early_stopping_enabled:
                    evals_without_improvement += 1
                    log.info(
                        "  No validation improvement for %d/%d evaluation(s) "
                        "(best=%.4f, min_delta=%.6f).",
                        evals_without_improvement,
                        early_stopping_patience,
                        best_val_loss,
                        early_stopping_min_delta,
                    )
                    if evals_without_improvement >= early_stopping_patience:
                        stop_training = True
                        early_stop_reason = (
                            "validation loss stopped improving"
                        )
                        log.info("  Early stopping triggered.")
                        if jlog is not None:
                            jlog.log(
                                "pretraining", "early_stopping",
                                step=completed_steps, epoch=eval_epoch,
                                val_loss=val_loss,
                                best_val_loss=(
                                    best_val_loss
                                    if best_val_loss != float("inf")
                                    else None
                                ),
                                patience=early_stopping_patience,
                                min_delta=early_stopping_min_delta,
                            )
                log.info("-" * 60)

            if distributed:
                control_code = 0
                if is_master:
                    if prune_this_step:
                        control_code = 1
                    elif stop_training:
                        control_code = 2
                flag = torch.tensor(control_code, device=device)
                dist.broadcast(flag, src=0)
                prune_this_step = flag.item() == 1
                stop_training = flag.item() == 2

            if prune_this_step:
                import optuna
                raise optuna.TrialPruned()

            if stop_training:
                early_stop_triggered = True
                break

            if (
                optimizer_updated
                and is_master
                and completed_steps > 0
                and completed_steps % tc["save_every"] == 0
            ):
                ckpt_path = checkpoint_manager.save_last(
                    model,
                    optimizer,
                    model_config_dict,
                    completed_steps,
                    eval_epoch,
                    scaler,
                    scheduler=scheduler,
                    trainer_state=trainer_state,
                )
                if ckpt_path is not None:
                    log.info(f"Saved checkpoint: {ckpt_path}")
                    if jlog is not None:
                        jlog.log(
                            "pretraining", "checkpoint",
                            step=completed_steps,
                            epoch=eval_epoch,
                            path=ckpt_path,
                            category="last",
                        )
                    try:
                        from pathlib import Path as _Path
                        from src.scripts.plot_training import plot_run_dir as _plot_run_dir
                        _plot_run_dir(_Path(run_dir))
                    except Exception as _exc:
                        log.debug("Auto-plotting skipped: %s", _exc)

            if is_epoch_boundary and completed_steps < tc["max_steps"]:
                data_iter = _build_epoch_iterator(
                    train_loader,
                    epoch=next_epoch,
                    train_sampler=train_sampler,
                    shuffle_generator=train_shuffle_generator,
                    data_seed=data_seed,
                )
            current_epoch = next_epoch
            epoch_micro_batches_consumed = next_micro_batch

    # final save
    final_epoch = next_epoch if next_micro_batch == 0 else current_epoch
    reported_epochs = (
        0
        if completed_steps == 0
        else next_epoch if next_micro_batch == 0 else current_epoch + 1
    )
    final_trainer_state = _checkpoint_trainer_state(
        next_epoch=next_epoch,
        next_micro_batch=next_micro_batch,
        data_seed=data_seed,
    )
    if is_master:
        final_ckpt = checkpoint_manager.save_last(
            model,
            optimizer,
            model_config_dict,
            completed_steps,
            final_epoch,
            scaler,
            scheduler=scheduler,
            trainer_state=final_trainer_state,
        )
        if final_ckpt is not None:
            log.info(f"Saved final checkpoint: {final_ckpt}")
            if jlog is not None:
                jlog.log(
                    "pretraining", "checkpoint",
                    step=completed_steps,
                    epoch=final_epoch,
                    path=final_ckpt,
                    category="last",
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
        if early_stop_triggered:
            log.info(
                "Stopped early after %d step(s): %s",
                completed_steps,
                early_stop_reason or "validation loss stopped improving",
            )
            if best_checkpoint_path is not None:
                log.info(f"Best checkpoint: {best_checkpoint_path}")
        log.info(fmt_training_complete(
            reported_epochs, completed_steps, total_hours, best_val_loss, run_dir,
        ))

    if jlog is not None:
        jlog.log(
            "pretraining", "training_complete",
            epochs=reported_epochs,
            total_steps=completed_steps,
            total_time_hours=round(total_hours, 2),
            best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
            best_checkpoint_path=best_checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            stopped_early=early_stop_triggered,
            stop_reason=early_stop_reason,
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
