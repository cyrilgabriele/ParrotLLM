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
    skip_ellipsis_filter: bool = False
    tokenizer_name: str = "openai-community/gpt2"
    append_eos_token: bool = False
    token_dtype: Literal["uint16", "uint32"] = "uint16"
    min_tokens: int = Field(default=64, ge=1)
    # Must match model.context_length in configs/default.yaml.  The training
    # data loader counts chunks via integer division, so any tail tokens that
    # don’t fill a complete window would be silently dropped.  Knowing the
    # context length at save time lets Phase 8 pad both splits to exact
    # multiples of (context_length + 1) and avoid this loss entirely.
    context_length: int = Field(default=1024, ge=1)

    # ── Token-budget subset download ───────────────────────────────────────────────
    # If set, preprocess.py will download/load a seeded random subset of OWT
    # sized to produce approximately `target_tokens` tokens after filtering.
    target_tokens: int | None = None
    subset_seed: int = 42

    # ── Topic classification (Phase 6.2) ─────────────────────────────────────────
    # Uses textattack/roberta-base-ag_news (4 classes: World, Sports, Business, Sci/Tech)
    # topic_classes: which labels to keep (None = skip topic filter entirely)
    # topic_distribution: optional per-class weights for resampling (None = keep all
    #   matching docs as-is). Weights are normalized if they do not sum to 1.0.
    topic_classes: list[str] | None = None
    topic_distribution: dict[str, float] | None = None
    skip_topic_filter: bool = False

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
        data = vars(args).copy()
        # Parse --topics tokens (e.g. ["Sports:0.4", "Business:0.6"]) into
        # topic_classes and topic_distribution before validation.
        raw_topics: list[str] | None = data.pop("topics", None)
        if raw_topics:
            classes, distribution = cls._parse_topics(raw_topics)
            data["topic_classes"] = classes
            data["topic_distribution"] = distribution
        return cls.model_validate(data)

    @staticmethod
    def _parse_topics(
        tokens: list[str],
    ) -> tuple[list[str], dict[str, float] | None]:
        """Parse 'Class' or 'Class:weight' tokens into (classes, distribution).

        Examples::

            _parse_topics(["Sports", "Business"])
            # -> (["Sports", "Business"], None)

            _parse_topics(["Sports:0.4", "Business:0.4", "World:0.2"])
            # -> (["Sports", "Business", "World"], {"Sports": 0.4, ...})
        """
        classes: list[str] = []
        weights: dict[str, float] = {}
        for token in tokens:
            if ":" in token:
                cls, w = token.rsplit(":", 1)
                cls = cls.strip()
                classes.append(cls)
                weights[cls] = float(w)
            else:
                classes.append(token.strip())

        # Warn if provided weights do not sum to ~1.0
        if weights:
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                warnings.warn(
                    f"Topic weights sum to {weight_sum:.3f} instead of 1.0; "
                    "they will be normalized automatically during resampling.",
                    stacklevel=3,
                )

        return classes, (weights if weights else None)
