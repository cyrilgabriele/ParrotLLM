"""Pydantic config for the SFT stage.

All defaults here are the ones recommended by VL07 and the docs/post_training/SFT.md
plan. They are safe to override per-experiment via the YAML config.

Field-by-field sourcing:

- ``hf_dataset_name``:  VL07 slide 48 — "Use Alpaca dataset for chat alignment
                        through SFT."
- ``learning_rate`` / ``min_lr``: VL07 slide 25 — "Use smaller learning rates"
                        as a catastrophic-forgetting mitigation. 2e-5 matches
                        Stanford Alpaca's published LR for LLaMA-7B; at our
                        40M scale 1e-5–5e-5 is a safe range.
- ``warmup_steps`` = 100: per §5 of SFT.md — short warmup because we are not
                        starting from random weights. See VL04 "Warmup
                        protects the start".
- ``epochs`` = 2:       SFT.md §5; Alpaca original used 3, we start at 2
                        because our model is much smaller and overfits
                        faster (VL07 slide 25 "avoid overtraining").
- ``weight_decay`` = 0.1: matches pretraining convention. VL07 slide 25
                        lists weight-decay regularisation as a CF mitigation.
- ``grad_clip`` = 1.0:  VL04 "Gradient Clipping" section — forces gradient
                        norm ≤ 1.0. Standard and cheap insurance against
                        exploding gradients during fine-tuning.
- ``pretraining_mix_ratio`` = 0.0: off by default. When > 0, a fraction of
                        each batch is drawn from the pretraining bin file
                        with ordinary next-token loss (no masking), as a
                        catastrophic-forgetting mitigation per VL07
                        slide 25 and SFT.md §3.4. Enable cautiously in
                        v2 after baseline SFT run is in.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # ── Data ────────────────────────────────────────────────────────────────
    hf_dataset_name: str = Field(
        default="tatsu-lab/alpaca",
        description=(
            "HF Hub dataset identifier. Default is Stanford Alpaca "
            "(VL07 slide 48 recommendation for PikoGPT)."
        ),
    )
    hf_split: str = Field(default="train")
    max_examples: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on number of examples. Useful for smoke tests "
            "(set to ~1000 to run an end-to-end test in seconds)."
        ),
    )
    val_fraction: float = Field(default=0.05, gt=0.0, lt=0.5)
    decontam_benchmarks: list[str] = Field(
        default_factory=list,
        description=(
            "HF Hub benchmark splits (e.g. 'lambada/test', 'hellaswag/validation') "
            "whose text is hashed and dropped from the SFT corpus. "
            "Fact sheet §4.3 requires this."
        ),
    )

    # ── Tokenisation / template ─────────────────────────────────────────────
    max_length: int = Field(
        default=1024,
        ge=64,
        description=(
            "Hard truncation ceiling. Must be ≤ model.context_length "
            "(1024 per PikoGPT fact sheet hard constraint)."
        ),
    )
    # Note: `append_eos` and `mask_instruction_loss` are intentionally NOT
    # configurable. Both are non-negotiable in the definition of SFT
    # (VL07 slide 15 + slide 14). Exposing them as YAML knobs misleads
    # readers into thinking the trainer respects them when in fact masking
    # is unconditional in SFTCollator and EOS is unconditional in
    # tokenise_example. If you need an ablation, hard-code the override
    # in trainer.py for the duration of the experiment.

    # ── Optimiser ───────────────────────────────────────────────────────────
    learning_rate: float = Field(default=2e-5, gt=0.0)
    min_lr: float = Field(default=2e-6, ge=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    beta2: float = Field(default=0.95, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)

    # ── Schedule ────────────────────────────────────────────────────────────
    warmup_steps: int = Field(default=100, ge=0)
    epochs: int = Field(default=2, ge=1)
    lr_schedule: str = Field(
        default="cosine",
        description="'cosine' or 'wsd'. Cosine matches Alpaca's published schedule.",
    )

    # ── Batching ────────────────────────────────────────────────────────────
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    num_workers: int = Field(default=2, ge=0)
    pin_memory: bool = Field(default=True)

    # ── Catastrophic-forgetting mitigations (VL07 slide 25) ─────────────────
    pretraining_mix_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of each batch drawn from pretraining tokens with "
            "ordinary next-token loss. 0.0 = pure SFT; 0.05–0.10 follows "
            "OLMo 2/3 practice. See SFT.md §3.4."
        ),
    )
    pretraining_bin_path: str | None = Field(
        default=None,
        description=(
            "Path to the pretraining train.bin (uint16 token array). "
            "Required only when pretraining_mix_ratio > 0."
        ),
    )

    # ── Logging / checkpointing ─────────────────────────────────────────────
    runs_dir: str = Field(default="runs")
    eval_every: int = Field(default=200, ge=1)
    save_every: int = Field(default=500, ge=1)
    log_every: int = Field(default=10, ge=1)
    early_stopping_patience: int = Field(default=5, ge=0)

    # ── Base checkpoint ─────────────────────────────────────────────────────
    base_checkpoint: str | None = Field(
        default=None,
        description=(
            "Path to the pretraining checkpoint to start from. Required at "
            "runtime; left optional in the schema so the CLI can override "
            "via --checkpoint."
        ),
    )

    # ── Device ──────────────────────────────────────────────────────────────
    device: str = Field(default="auto")

    @field_validator("lr_schedule")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in {"cosine", "wsd"}:
            raise ValueError("sft.lr_schedule must be 'cosine' or 'wsd'.")
        return v
