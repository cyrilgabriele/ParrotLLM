"""Project-level configuration aggregation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from .preprocessing.preprocessConfig import PreprocessConfig
from .training.trainingConfig import HfUploadConfig, ModelConfig, TrainingConfig
from .tuning.tuneConfig import TuneConfig
from .loggingConfig import LoggingConfig
from .post_training.sftConfig import SFTConfig
from .post_training.dpoConfig import DPOConfig


class EvalDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    path: str
    subset: str | None = None
    split: str | None = None


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(...)
    batch_size: PositiveInt = Field(...)
    max_sequences: PositiveInt = Field(...)
    datasets: list[EvalDatasetConfig] = Field(default_factory=list)


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(...)
    max_tokens: PositiveInt = Field(...)
    temperature: float = Field(..., ge=0.0)
    top_k: PositiveInt = Field(...)
    top_p: float = Field(..., gt=0.0, le=1.0)
    # VL09 slide 25 — θ=1.0 disables; PikoGPT default 1.1.
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)


class ChatConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(...)
    max_tokens: PositiveInt = Field(...)
    temperature: float = Field(..., ge=0.0)
    top_k: PositiveInt = Field(...)
    top_p: float = Field(..., gt=0.0, le=1.0)
    # VL09 slide 30 "PikoGPT default": θ=1.1.
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    system_prompt: str = Field(...)
    checkpoint_dir: Path = Field(...)

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def _coerce_checkpoint_dir(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))


class ProjectConfig(BaseModel):
    """Typed view over the entire project configuration tree."""

    model_config = ConfigDict(extra="ignore")

    preprocess: PreprocessConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    tune: TuneConfig | None = None
    logging: LoggingConfig | None = None
    eval: EvalConfig | None = None
    inference: InferenceConfig | None = None
    chat: ChatConfig | None = None
    sft: SFTConfig | None = None  # VL07 post-training stage (see configs/post_training/sftConfig.py)
    dpo: DPOConfig | None = None  # VL08 post-training stage (see configs/post_training/dpoConfig.py)


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate the full project configuration from YAML."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = yaml.safe_load(path.read_text())
    if payload is None:
        raise ValueError(f"Config file {path} is empty.")
    if not isinstance(payload, Dict):
        raise TypeError(f"Config file {path} must define a mapping at the top level.")
    return ProjectConfig.model_validate(payload)


def load_project_config_from_checkpoint(checkpoint_path: str | Path) -> ProjectConfig:
    """Load and validate the saved project configuration from a checkpoint."""

    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Dict):
        raise TypeError(
            f"Checkpoint {path} must contain a mapping payload, got {type(checkpoint).__name__}."
        )

    payload = checkpoint.get("config")
    if payload is None:
        raise ValueError(f"Checkpoint {path} does not contain a saved config.")
    if not isinstance(payload, Dict):
        raise TypeError(
            f"Checkpoint {path} must store 'config' as a mapping, got {type(payload).__name__}."
        )
    return ProjectConfig.model_validate(payload)


__all__ = [
    "ChatConfig",
    "EvalConfig",
    "EvalDatasetConfig",
    "HfUploadConfig",
    "DPOConfig",
    "InferenceConfig",
    "ProjectConfig",
    "SFTConfig",
    "load_project_config",
    "load_project_config_from_checkpoint",
]
