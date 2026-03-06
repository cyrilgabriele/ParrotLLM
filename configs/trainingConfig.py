"""Pydantic training & logging configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    console_level: str = "INFO"
    file_level: str = "DEBUG"
    components: dict[str, str] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    vocab_size: int = 50258
    pad_token_id: int = 50257
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    d_model: int = 320
    n_layers: int = 16
    n_heads: int = 8
    d_ff: int = 854
    context_length: int = 1024
    bias: bool = False
    dropout: float = 0.0
    rope_theta: float = 10000.0


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed: int = 42
    device: str = "auto"

    # data
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"

    # batching
    batch_size: int = 64
    gradient_accumulation_steps: int = 4

    # optimizer
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # schedule
    warmup_steps: int = 2000
    max_steps: int = 100000

    # checkpointing
    save_every: int = 5000
    eval_every: int = 500
    checkpoint_dir: str = "checkpoints"

    # logging
    runs_dir: str = "runs"
    log_every: int = 10
