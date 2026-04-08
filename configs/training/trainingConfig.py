"""Pydantic training & logging configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class HfUploadConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    repo_id: str = Field(..., description="Target Hugging Face repo in owner/name form.")
    repo_type: Literal["model", "dataset", "space"] = Field(default="model")
    path_in_repo: str = Field(
        default="",
        description=(
            "Optional prefix inside the Hub repo. The local run path relative to the "
            "project root is appended under this prefix."
        ),
    )
    private: bool | None = Field(
        default=None,
        description="Privacy used only when auto-creating the target repo.",
    )

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, value: str) -> str:
        repo_id = value.strip()
        if not repo_id:
            raise ValueError("training.hf_upload.repo_id must not be empty.")
        return repo_id

    @field_validator("path_in_repo")
    @classmethod
    def _normalise_path_in_repo(cls, value: str) -> str:
        return str(value).strip().replace("\\", "/").strip("/")


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
    train_samples_per_epoch: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional number of random overlapping training windows to draw per epoch. "
            "Defaults to the legacy non-overlapping chunk count."
        ),
    )
    val_sequence_stride: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional stride for deterministic validation windows. "
            "Defaults to model.context_length."
        ),
    )

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
    hf_upload: HfUploadConfig | None = Field(
        default=None,
        description=(
            "Optional Hugging Face upload target. When set, the finished run "
            "directory is mirrored to the Hub once at the end of training."
        ),
    )

    # torch.compile toggle (disable for short HP tuning trials)
    compile: bool = Field(True)

    @model_validator(mode="after")
    def _validate_scheduler_bounds(self) -> "TrainingConfig":
        if self.min_lr > self.learning_rate:
            raise ValueError("training.min_lr must not exceed training.learning_rate.")
        return self
