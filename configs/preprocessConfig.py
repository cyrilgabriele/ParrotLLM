"""Pydantic-based configuration objects used across the project."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_LANG = "en"

# Valid topic class names for textattack/roberta-base-ag_news
VALID_TOPIC_CLASSES = frozenset({"World", "Sports", "Business", "Sci/Tech"})


class PreprocessConfig(BaseModel):
    """Validated configuration for the preprocessing pipeline."""
    dataset_size: Literal["small", "full", "dummy"] = Field(
        ...,
        description="Which OpenWebText subset to load (dummy=100 docs, small=10k, full=entire set).",
    )
    lang: str = Field(..., min_length=1, description="Target ISO language code (e.g. en).")
    data_dir: Path = Field(..., description="Root directory containing downloaded datasets.")
    num_workers: int | Literal["auto"] = Field(..., description="Number of worker processes or 'auto'.")
    skip_dedup: bool = Field(..., description="Whether to skip MinHash deduplication (Phase 6).")
    skip_decontam: bool = Field(..., description="Whether to skip decontamination (Phase 1).")
    filter_mode: Literal["none", "heuristic", "classifier"] = Field(
        ..., description="Filtering mode for code/quality phases.")
    skip_code_filter: bool = Field(..., description="Skip Phase 4 code/artifact filtering.")
    skip_quality_filter: bool = Field(..., description="Skip Phase 5 quality filtering.")
    skip_ellipsis_filter: bool = Field(..., description="Skip Phase 6.1 ellipsis filtering.")
    append_eos_token: bool = Field(..., description="Append EOS token per document before saving.")
    token_dtype: Literal["uint16", "uint32"] = Field(..., description="Output dtype for binary files.")
    min_tokens: int = Field(..., ge=1, description="Minimum tokens required to keep a document.")
    # Must match model.context_length in configs/default.yaml.  The training
    # data loader counts chunks via integer division, so any tail tokens that
    # don’t fill a complete window would be silently dropped.  Knowing the
    # context length at save time lets Phase 8 pad both splits to exact
    # multiples of (context_length + 1) and avoid this loss entirely.
    context_length: int = Field(default=1024, ge=1)

    # ── Token-budget subset download ───────────────────────────────────────────────
    # If set, preprocess.py will download/load a seeded random subset of OWT
    # sized to produce approximately `target_tokens` tokens after filtering.
    target_tokens: int | None = Field(
        default=None,
        description="Optional token budget for subset download.",
    )

    # ── Topic classification (Phase 6.2) ─────────────────────────────────────────
    # Uses textattack/roberta-base-ag_news (4 classes: World, Sports, Business, Sci/Tech)
    # topic_classes: which labels to keep (None = skip topic filter entirely)
    # topic_distribution: optional per-class weights for resampling (None = keep all
    #   matching docs as-is). Weights are normalized if they do not sum to 1.0.
    topic_classes: list[str] | None = Field(
        default=None,
        description="Subset of AG News classes to keep; None disables topic filtering.",
    )
    topic_distribution: dict[str, float] | None = Field(
        default=None,
        description="Optional per-class resampling weights.",
    )
    skip_topic_filter: bool = Field(..., description="Force-skip topic filtering even if classes provided.")

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

    @field_validator("topic_classes")
    @classmethod
    def _validate_topic_classes(cls, value: list[str] | None) -> list[str] | None:
        if not value:
            return None
        invalid = sorted(set(value) - VALID_TOPIC_CLASSES)
        if invalid:
            raise ValueError(
                f"topic_classes contains invalid labels: {', '.join(invalid)}; "
                f"valid classes are {', '.join(sorted(VALID_TOPIC_CLASSES))}."
            )
        return value

    @field_validator("topic_distribution")
    @classmethod
    def _normalize_topic_distribution(
        cls, value: dict[str, float] | None
    ) -> dict[str, float] | None:
        if not value:
            return None
        for key in value:
            if key not in VALID_TOPIC_CLASSES:
                raise ValueError(
                    f"topic_distribution contains invalid label '{key}'. Valid options: "
                    f"{', '.join(sorted(VALID_TOPIC_CLASSES))}."
                )
        weight_sum = sum(value.values())
        if weight_sum <= 0:
            raise ValueError("topic_distribution weights must sum to a positive value.")
        if abs(weight_sum - 1.0) > 0.01:
            warnings.warn(
                f"Topic weights sum to {weight_sum:.3f} instead of 1.0; they will be normalized.",
                stacklevel=3,
            )
        return value
