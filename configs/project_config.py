"""Project-level configuration aggregation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from .preprocessConfig import PreprocessConfig
from .trainingConfig import ModelConfig, TrainingConfig
from .tuneConfig import TuneConfig
from .loggingConfig import LoggingConfig


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


class ChatConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str = Field(...)
    max_tokens: PositiveInt = Field(...)
    temperature: float = Field(..., ge=0.0)
    top_k: PositiveInt = Field(...)
    top_p: float = Field(..., gt=0.0, le=1.0)
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


__all__ = [
    "ChatConfig",
    "EvalConfig",
    "EvalDatasetConfig",
    "InferenceConfig",
    "ProjectConfig",
    "load_project_config",
]
