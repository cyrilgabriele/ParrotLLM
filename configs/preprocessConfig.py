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

    # ── Tokenizer ──────────────────────────────────────────────────────────────────
    tokenizer_name: str | None = Field(
        default=None,
        description="Tokenizer model name (e.g. 'openai-community/gpt2'). If None, uses DEFAULT_TOKENIZER_NAME.",
    )

    # ── Deduplication (Phase 6) ────────────────────────────────────────────────────
    dedup_num_perm: int = Field(
        default=16,
        ge=1,
        description="MinHash permutations per document.",
    )
    dedup_bands: int = Field(
        default=4,
        ge=1,
        description="Number of LSH bands; must satisfy bands × rows == num_perm.",
    )
    dedup_rows: int = Field(
        default=4,
        ge=1,
        description="Rows per LSH band.",
    )
    dedup_shingle_size: int = Field(
        default=5,
        ge=1,
        description="Word n-gram window for deduplication fingerprinting.",
    )
    dedup_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity threshold for near-duplicates.",
    )

    # ── Language & Topic Detection (Phases 3, 3.5) ──────────────────────────────────
    language_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum fastText confidence to keep a document for target language.",
    )
    topic_model_name: str = Field(
        default="textattack/distilbert-base-uncased-ag-news",
        description="Hugging Face model name for AG News topic classification.",
    )
    topic_text_truncation: int = Field(
        default=256,
        ge=1,
        description="Characters to truncate per document in topic classification.",
    )
    topic_batch_size: int = Field(
        default=512,
        ge=1,
        description="Batch size for topic classifier.",
    )

    # ── Heuristic Filter Thresholds (Phases 2, 4, 5) ───────────────────────────────
    html_tag_density_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Max fraction of characters that may be HTML/XML tags.",
    )
    code_symbol_ratio_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Max allowed fraction of code-symbol characters.",
    )
    programming_keyword_max: int = Field(
        default=3,
        ge=0,
        description="Max distinct programming keywords before doc is flagged as code.",
    )
    min_word_count: int = Field(
        default=50,
        ge=1,
        description="Minimum words per document.",
    )
    min_char_count: int = Field(
        default=200,
        ge=1,
        description="Minimum characters per document.",
    )
    max_word_length: int = Field(
        default=40,
        ge=1,
        description="Max word length (longer tokens likely URLs or hashes).",
    )
    ngram_size: int = Field(
        default=10,
        ge=1,
        description="Sliding-window size for n-gram repetition detection.",
    )
    ngram_max_repeats: int = Field(
        default=3,
        ge=1,
        description="Max allowed repetitions of any n-gram.",
    )

    # ── Ellipsis Filter (Phase 6.1) ────────────────────────────────────────────────
    ellipsis_ratio_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Max ellipsis count per word; exceeding drops the doc.",
    )

    # ── Classifier Thresholds (Phases 4, 5) ───────────────────────────────────────
    code_classifier_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum fastText confidence to classify doc as code.",
    )
    kenlm_perplexity_low: float = Field(
        default=10.0,
        ge=0.0,
        description="KenLM perplexity floor; below this is unnaturally repetitive.",
    )
    kenlm_perplexity_high: float = Field(
        default=100000.0,
        ge=0.0,
        description="KenLM perplexity ceiling; above this is incoherent.",
    )
    educational_quality_min: int = Field(
        default=2,
        ge=0,
        le=5,
        description="fastText edu-quality min (0-5 scale).",
    )

    # ── Processing Parameters ──────────────────────────────────────────────────────
    batch_size: int = Field(
        default=2048,
        ge=1,
        description="Batch size for dataset.map operations.",
    )
    minimum_tokens_per_doc: int = Field(
        default=64,
        ge=1,
        description="Minimum tokens per document after tokenization (Phase 7).",
    )
    validation_split_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of tokens reserved for validation (Phase 8).",
    )
    num_workers_multiplier: float = Field(
        default=2.0,
        ge=0.1,
        description="When num_workers='auto', multiply cpu_count by this value.",
    )

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
