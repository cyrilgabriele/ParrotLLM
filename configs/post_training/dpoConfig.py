"""Pydantic config for the DPO stage (VL08).

DPO (Direct Preference Optimization, Rafailov et al. 2023, arXiv:2305.18290)
is the third post-training stage in the VL07/VL08 pipeline:

    base → SFT → DPO

It teaches the model to PREFER chosen responses over rejected ones,
relative to the original SFT model's distribution. Inputs are
preference pairs (prompt, chosen, rejected). Two models live in memory:
- Policy: the model being trained, initialised from the SFT checkpoint.
- Reference: a frozen copy of the same SFT checkpoint, used to compute
  the implicit reward via log-prob ratios.

The Alpaca template (slide 32) is reused unchanged so prompts that
worked at SFT time work the same at DPO time and at inference.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DPOConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # ── Data ────────────────────────────────────────────────────────────────
    hf_dataset_name: str = Field(
        default="Intel/orca_dpo_pairs",
        description=(
            "HF preference dataset. Default Intel/orca_dpo_pairs (12k pairs, "
            "GPT-4 chosen vs original-LLaMA rejected, Alpaca-friendly). "
            "Alternatives: HuggingFaceH4/ultrafeedback_binarized (60k, harder), "
            "argilla/distilabel-intel-orca-dpo-pairs (cleaned variant)."
        ),
    )
    hf_split: str = Field(default="train")
    max_examples: int | None = Field(
        default=None, ge=1,
        description="Cap on examples (e.g. 100 for smoke tests).",
    )
    val_fraction: float = Field(default=0.05, gt=0.0, lt=0.5)
    decontam_benchmarks: list[str] = Field(
        default_factory=list,
        description="Same registry as SFT (lambada/hellaswag/winogrande/openbookqa).",
    )
    preference_jsonl_path: str | None = Field(
        default=None,
        description=(
            "Optional local JSONL of {prompt, chosen, rejected} preference "
            "pairs. Rows may set template='raw' for leaderboard-style prompts."
        ),
    )
    preference_oversample: int = Field(
        default=1,
        ge=1,
        description="Repeat factor for local preference JSONL rows.",
    )
    hf_cache_dir: str | None = Field(
        default=None,
        description=(
            "Optional datasets.load_dataset cache_dir override. Leave unset "
            "to use Hugging Face defaults."
        ),
    )
    cleanup_hf_cache: bool = Field(
        default=True,
        description=(
            "Delete Hugging Face dataset caches after DPO/decontam data has "
            "been materialized. Keeps Mac storage from filling up during "
            "repeated post-training runs."
        ),
    )

    # ── Tokenisation ────────────────────────────────────────────────────────
    max_length: int = Field(
        default=1024, ge=64,
        description="Hard truncation ceiling. ≤ model.context_length.",
    )

    # ── DPO loss hyperparameters (Rafailov et al. 2023) ─────────────────────
    beta: float = Field(
        default=0.1, gt=0.0,
        description=(
            "DPO temperature. Higher β = more conservative (stays closer to "
            "the reference). Typical 0.1-0.5; 0.1 is the canonical default. "
            "At small model scale push toward 0.3+ if the policy drifts too far."
        ),
    )
    length_normalize_logp: bool = Field(
        default=False,
        description=(
            "Divide log-probs by response length before the DPO contrast. "
            "Reduces the bias toward shorter responses; off by default per "
            "the original DPO paper, but worth flipping if you observe a "
            "length-collapse failure mode at small scale."
        ),
    )

    # ── Optimiser ───────────────────────────────────────────────────────────
    learning_rate: float = Field(default=5.0e-6, gt=0.0,
                                 description="DPO LR is typically 1/2-1/4 of SFT LR.")
    min_lr: float = Field(default=5.0e-7, ge=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0,
                                description="Standard DPO recipe: no weight decay.")
    beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    beta2: float = Field(default=0.95, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0)

    # ── Schedule ────────────────────────────────────────────────────────────
    warmup_steps: int = Field(default=50, ge=0)
    epochs: int = Field(default=1, ge=1,
                        description="DPO usually 1 epoch (Rafailov + Tunstall et al.).")
    lr_schedule: str = Field(default="cosine")

    # ── Batching ────────────────────────────────────────────────────────────
    batch_size: int = Field(default=4, ge=1,
                            description="Smaller than SFT — DPO holds 2 models + 4 forwards/step.")
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    loss_chunk_rows: int = Field(
        default=2048,
        ge=128,
        description="Rows used for chunked response-token log-prob scoring.",
    )
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=True)

    # ── CF mitigations (same registry as SFT) ───────────────────────────────
    wt103_eval_every_n_evals: int = Field(default=2, ge=0)
    wt103_max_sequences: int = Field(default=64, ge=1)
    wt103_hard_stop_pct: float = Field(default=10.0, ge=0.0)

    # ── Logging / checkpointing ─────────────────────────────────────────────
    runs_dir: str = Field(default="runs")
    eval_every: int = Field(default=50, ge=1)
    save_every: int = Field(default=999999, ge=1)
    log_every: int = Field(default=5, ge=1)
    early_stopping_patience: int = Field(default=3, ge=0)

    # ── Base SFT checkpoint ─────────────────────────────────────────────────
    base_checkpoint: str | None = Field(
        default=None,
        description=(
            "Path to the SFT checkpoint. DPO uses it as BOTH the policy "
            "init AND the frozen reference. CLI --checkpoint overrides."
        ),
    )

    # ── Device / perf ───────────────────────────────────────────────────────
    device: str = Field(default="auto")
    torch_compile: bool = Field(default=True)

    @field_validator("lr_schedule")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        v = value.strip().lower()
        if v not in {"cosine", "wsd"}:
            raise ValueError("dpo.lr_schedule must be 'cosine' or 'wsd'.")
        return v
