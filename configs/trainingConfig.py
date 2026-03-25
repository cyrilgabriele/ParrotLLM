"""Pydantic training & logging configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(...)

    # data
    train_bin: str = Field(...)
    val_bin: str = Field(...)

    # batching
    batch_size: int = Field(...)
    gradient_accumulation_steps: int = Field(...)

    # optimizer
    learning_rate: float = Field(...)
    min_lr: float = Field(...)
    weight_decay: float = Field(...)
    beta1: float = Field(...)
    beta2: float = Field(...)
    grad_clip: float = Field(...)

    # schedule
    warmup_steps: int = Field(...)
    max_steps: int = Field(...)
    # "wsd" = Warmup-Stable-Decay (linear decay-to-zero); "cosine" = cosine annealing
    lr_schedule: str = Field(...)
    # Fraction of max_steps used for the decay phase (WSD only)
    lr_decay_ratio: float = Field(...)

    # z-loss coefficient; 1e-4 is standard, 0 disables z-loss
    z_loss_coeff: float = Field(...)

    # checkpointing
    save_every: int = Field(...)
    eval_every: int = Field(...)
    checkpoint_dir: str = Field(...)

    # logging
    runs_dir: str = Field(...)
    log_every: int = Field(...)
