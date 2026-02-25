"""Pydantic-based configuration objects used across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_LANG = "en"


class PreprocessConfig(BaseModel):
    """Validated configuration for the preprocessing pipeline."""

    model_config = ConfigDict(extra="ignore")

    dataset_size: Literal["small", "full", "dummy"] = "full"
    lang: str = Field(default=DEFAULT_LANG, min_length=1)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    num_workers: int | Literal["auto"] = "auto"
    skip_dedup: bool = False
    skip_decontam: bool = False
    filter_mode: Literal["none", "heuristic", "classifier"] = "heuristic"
    skip_code_filter: bool = False
    skip_quality_filter: bool = False
    tokenizer_name: str = "openai-community/gpt2"
    append_eos_token: bool = False
    token_dtype: Literal["uint16", "uint32"] = "uint16"
    min_tokens: int = Field(default=64, ge=1)

    @field_validator("data_dir", mode="before")
    @classmethod
    def _coerce_data_dir(cls, value):
        if isinstance(value, Path):
            return value
        return Path(str(value))

    @field_validator("num_workers", mode="before")
    @classmethod
    def _coerce_num_workers(cls, value):
        if value is None:
            return "auto"
        if isinstance(value, str):
            value = value.strip()
            if value == "auto" or value == "":
                return "auto"
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_workers must be 'auto' or a positive integer") from exc
        if int_value < 1:
            raise ValueError("num_workers must be >= 1")
        return int_value

    @classmethod
    def from_args(cls, args) -> "PreprocessConfig":
        """Build config from an argparse namespace."""
        return cls.model_validate(vars(args))
