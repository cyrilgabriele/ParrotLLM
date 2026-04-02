"""Pydantic training & logging configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    vocab_size: int = Field(...)
    pad_token_id: int = Field(...)
    bos_token_id: int = Field(...)
    eos_token_id: int = Field(...)
    d_model: int = Field(...)
    n_layers: int = Field(...)
    n_heads: int = Field(...)
    d_ff: int = Field(...)
    context_length: int = Field(...)
    bias: bool = Field(...)
    dropout: float = Field(...)
    rope_theta: float = Field(...)
    gradient_checkpointing: bool = Field(False)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(...)

    # data
    train_bin: str = Field(...)
    val_bin: str = Field(...)
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = Field(True)

    # batching
    batch_size: int = Field(...)
    gradient_accumulation_steps: int = Field(...)

    # optimizer
    learning_rate: float = Field(..., gt=0.0)
    min_lr: float = Field(..., ge=0.0)
    weight_decay: float = Field(...)
    beta1: float = Field(...)
    beta2: float = Field(...)
    grad_clip: float = Field(...)

    # schedule
    warmup_steps: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    # "wsd" = Warmup-Stable-Decay (linear decay-to-zero); "cosine" = cosine annealing
    lr_schedule: Literal["wsd", "cosine"] = Field(...)
    # Fraction of max_steps used for the decay phase (WSD only)
    lr_decay_ratio: float = Field(..., ge=0.0, le=1.0)

    # z-loss coefficient; 1e-4 is standard, 0 disables z-loss
    z_loss_coeff: float = Field(...)

    # checkpointing
    save_every: int = Field(...)
    eval_every: int = Field(...)
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Checkpoint subdirectory inside the per-run runs_dir/run_* folder.",
    )
    keep_last_checkpoints: int = Field(10, ge=0)
    keep_best_checkpoints: int = Field(10, ge=0)
    early_stopping_patience: int = Field(0, ge=0)
    early_stopping_min_delta: float = Field(0.0, ge=0.0)

    # logging
    runs_dir: str = Field(...)
    log_every: int = Field(...)

    # torch.compile toggle (disable for short HP tuning trials)
    compile: bool = Field(True)

    @model_validator(mode="after")
    def _validate_scheduler_bounds(self) -> "TrainingConfig":
        if self.min_lr > self.learning_rate:
            raise ValueError("training.min_lr must not exceed training.learning_rate.")
        return self
