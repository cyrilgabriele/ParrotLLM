"""Typed configuration for supervised fine-tuning (SFT) posttraining."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SFTSourceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(...)
    loader: Literal[
        "ai2_arc",
        "alpaca",
        "wildchat",
        "oasst1",
        "tulu",
        "wildguardmix",
        "pku_safe_rlhf_qa",
        "local_jsonl",
        "sciq",
        "commonsense_qa",
        "cosmos_qa",
        "social_iqa",
        "race",
        "mmlu",
        "boolq",
        "piqa",
        "wsc273",
        "hellaswag",
        "winogrande",
        "openbookqa",
        "narrative_completion",
        "cbt",
    ] = Field(...)
    path: str = Field(...)
    subset: str | None = Field(default=None)
    split: str = Field(default="train")
    target_examples: int = Field(..., ge=1)
    candidate_multiplier: int = Field(default=4, ge=1)
    source_matches: list[str] = Field(default_factory=list)
    language: str | None = Field(default="en")
    require_model_substring: str | None = Field(default=None)
    exclude_toxic: bool = Field(default=True)
    exclude_redacted: bool = Field(default=True)
    keep_harmful_only: bool = Field(default=False)
    use_best_branch: bool = Field(default=False)
    require_tree_state: str | None = Field(default=None)
    min_turns: int = Field(default=2, ge=1)
    max_turns: int = Field(default=6, ge=1)
    max_depth: int = Field(default=4, ge=1)
    drop_chain_of_thought: bool = Field(default=True)
    quality_weight: float = Field(default=1.0, gt=0.0)
    tags: list[str] = Field(default_factory=list)
    rationale: str = Field(default="")
    # None = inherit SFTConfig.template_format (the global default).
    # "raw" lets a single source (e.g. narrative_completion for LAMBADA) skip the
    # Alpaca chat wrapper while the rest of the mix stays alpaca-wrapped.
    template_format: Literal["alpaca", "raw"] | None = Field(default=None)


class SFTDecontamConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(...)
    loader: Literal["local_disk", "huggingface"] = Field(...)
    path: str = Field(...)
    subset: str | None = Field(default=None)
    split: str = Field(default="test")
    field: str | None = Field(default=None)
    enabled: bool = Field(default=True)


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(default="auto")
    base_checkpoint: Path = Field(...)
    cache_dir: Path | None = Field(default=Path("data/posttraining/hf_cache"))
    raw_dir: Path = Field(default=Path("data/posttraining/raw"))
    prepared_dir: Path = Field(default=Path("data/posttraining/sft_mix"))
    runs_dir: Path = Field(default=Path("runs/posttraining/sft"))
    checkpoint_dir: str = Field(default="checkpoints")
    system_prompt: str = Field(default="You are ParrotLLM, a helpful assistant.")
    template_format: Literal["alpaca", "raw"] = Field(default="alpaca")
    max_seq_length: int = Field(default=1024, ge=32)
    train_batch_size: int = Field(default=8, ge=1)
    eval_batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rates: list[float] = Field(default_factory=lambda: [5e-5, 1e-4, 2e-4])
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    beta2: float = Field(default=0.95, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    z_loss_coeff: float = Field(default=0.0, ge=0.0)
    replay_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    replay_train_bin: Path = Field(default=Path("data/processed/train.bin"))
    replay_val_bin: Path = Field(default=Path("data/processed/val.bin"))
    num_epochs: float = Field(default=1.0, gt=0.0)
    polish_epochs: float = Field(default=0.25, ge=0.0)
    polish_subset_size: int = Field(default=4000, ge=1)
    save_every: int = Field(default=250, ge=1)
    eval_every: int = Field(default=100, ge=1)
    early_stopping_metric: str = Field(default="composite_score")
    early_stopping_mode: Literal["min", "max"] = Field(default="min")
    early_stopping_target: float | None = Field(default=None)
    early_stopping_patience: int = Field(default=0, ge=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    keep_last_checkpoints: int = Field(default=3, ge=0)
    keep_best_checkpoints: int = Field(default=2, ge=0)
    log_every: int = Field(default=10, ge=1)
    seed: int = Field(default=42)
    compile: bool = Field(default=False)
    format_score_weight: float = Field(default=0.1, ge=0.0)
    forgetting_penalty_weight: float = Field(default=0.05, ge=0.0)
    prompt_suite_path: Path | None = Field(default=Path("configs/posttraining/dev_prompt_suite.jsonl"))
    log_prompt_suite_generations: bool = Field(default=True)
    sources: list[SFTSourceConfig] = Field(default_factory=list)
    decontam_datasets: list[SFTDecontamConfig] = Field(default_factory=list)

    @field_validator(
        "base_checkpoint",
        "cache_dir",
        "raw_dir",
        "prepared_dir",
        "runs_dir",
        "replay_train_bin",
        "replay_val_bin",
        "prompt_suite_path",
        mode="before",
    )
    @classmethod
    def _coerce_paths(cls, value):
        if value is None or isinstance(value, Path):
            return value
        return Path(str(value))


__all__ = ["SFTConfig", "SFTDecontamConfig", "SFTSourceConfig"]
