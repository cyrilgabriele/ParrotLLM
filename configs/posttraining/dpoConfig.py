"""Typed configuration for Direct Preference Optimization (DPO) posttraining."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DPOSourceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(...)
    path: str = Field(...)               # HF dataset id, e.g. "Anthropic/hh-rlhf"
    subset: str | None = Field(default=None)
    split: str = Field(default="train")
    target_pairs: int = Field(..., ge=1)
    language: str | None = Field(default="en")
    # SFT loader name (e.g. "piqa", "sciq"); required when DPOConfig.preference_format == "mc_letter".
    loader: str | None = Field(default=None)


class DPOConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(default="auto")
    reference_checkpoint: Path = Field(...)  # path to the SFT checkpoint
    cache_dir: Path | None = Field(default=Path("data/posttraining/hf_cache"))
    raw_dir: Path = Field(default=Path("data/posttraining/dpo_raw"))
    prepared_dir: Path = Field(default=Path("data/posttraining/dpo_pairs"))
    runs_dir: Path = Field(default=Path("runs/posttraining/dpo"))
    checkpoint_dir: str = Field(default="checkpoints")
    system_prompt: str = Field(default="You are ParrotLLM, a helpful assistant.")
    max_seq_length: int = Field(default=1024, ge=32)

    # Hyperparameters (per spec, Phase 1).
    beta: float = Field(default=0.1, gt=0.0)
    learning_rate: float = Field(default=5.0e-7, gt=0.0)
    num_epochs: float = Field(default=1.0, gt=0.0)
    train_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)  # MPS bug -> must stay 1
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    beta2: float = Field(default=0.999, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    seed: int = Field(default=42)

    # Logging / eval cadence.
    save_every: int = Field(default=200, ge=1)
    eval_every: int = Field(default=100, ge=1)
    log_every: int = Field(default=1, ge=1)
    keep_last_checkpoints: int = Field(default=2, ge=1)
    keep_best_checkpoints: int = Field(default=2, ge=1)

    # Data.
    preference_format: Literal["hh_rlhf", "mc_letter"] = Field(default="hh_rlhf")
    sources: list[DPOSourceConfig] = Field(default_factory=list)
    decontam_datasets: list = Field(default_factory=list)
    dev_pairs: int = Field(default=500, ge=1)


__all__ = ["DPOConfig", "DPOSourceConfig"]
