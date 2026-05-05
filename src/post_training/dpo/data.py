"""DPO data pipeline: load preference pairs, normalise, render, tokenise, split.

Mirrors `src/post_training/sft/data.py` in shape so the trainer can use
the same patterns. The five logical steps:

1. Load from HF Hub (default `Intel/orca_dpo_pairs` — 12k Alpaca-style
   pairs of GPT-4 vs original-LLaMA responses; small enough that our
   35M-param model can iterate without long wall-clocks).
2. Normalise to internal schema ``{prompt, chosen, rejected}``.
3. Decontaminate against the leaderboard test splits — reused from
   `src.post_training.sft.data` so SFT and DPO drop identical strings.
4. Render prompts with the Alpaca template (slide 32 critical rule —
   same template SFT was trained on), tokenise both completions
   separately, record per-completion prompt_length.
5. Split train/val (default 5%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.post_training.dpo.template import DPO_DEFAULT_TEMPLATE
from src.post_training.sft.data import (
    build_decontam_index,
    filter_contaminated,
)


log = logging.getLogger("parrotllm.dpo.data")


# ── Schema normalisation ─────────────────────────────────────────────────────

def normalise_dpo_example(raw: dict) -> dict:
    """Map a raw HF preference row to ``{prompt, chosen, rejected}``.

    Handles the schemas seen in:
    - ``Intel/orca_dpo_pairs`` — ``{system?, question, chosen, rejected}``
    - ``HuggingFaceH4/ultrafeedback_binarized`` — ``{prompt, chosen: [{role,content}], rejected: [...]}``
    - ``argilla/*`` — same as ultrafeedback
    - generic ``{prompt, chosen, rejected}`` strings

    Raises ValueError on anything unrecognised so the caller gets a hard
    failure at data-load time rather than silent mistraining.
    """
    # Variant 1: orca-style { question, chosen, rejected } ± system prompt.
    if "question" in raw and "chosen" in raw and "rejected" in raw:
        prompt = raw["question"]
        if raw.get("system"):
            prompt = f"{raw['system']}\n\n{prompt}"
        return {
            "prompt": prompt,
            "chosen": _stringify_completion(raw["chosen"]),
            "rejected": _stringify_completion(raw["rejected"]),
        }

    # Variant 2 & 3: ultrafeedback / generic { prompt, chosen, rejected }.
    if "prompt" in raw and "chosen" in raw and "rejected" in raw:
        return {
            "prompt": _stringify_completion(raw["prompt"]),
            "chosen": _stringify_completion(raw["chosen"]),
            "rejected": _stringify_completion(raw["rejected"]),
        }

    raise ValueError(f"Unrecognised DPO schema: keys={list(raw.keys())}")


def _stringify_completion(value) -> str:
    """Coerce HF chat-format lists or plain strings to a single string.

    UltraFeedback stores ``chosen`` as ``[{"role": "user", "content": ...},
    {"role": "assistant", "content": ...}]`` — for our prompt template we
    extract the LAST assistant message (or just concat user when present).
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        # Pick the last assistant turn if any; fall back to concat.
        for msg in reversed(value):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
        return "\n".join(
            str(m.get("content", "")) for m in value if isinstance(m, dict)
        ).strip()
    return str(value).strip()


# ── Tokenisation ─────────────────────────────────────────────────────────────

@dataclass
class TokenisedDPOExample:
    """One fully-tokenised DPO preference pair.

    chosen_input_ids and rejected_input_ids both contain the SAME prompt
    prefix followed by their respective completions and an EOS. The
    prompt_lengths match for both halves (same instruction text).
    """

    chosen_input_ids: list[int]
    chosen_prompt_length: int
    rejected_input_ids: list[int]
    rejected_prompt_length: int

    def to_dict(self) -> dict:
        return {
            "chosen_input_ids": self.chosen_input_ids,
            "chosen_prompt_length": self.chosen_prompt_length,
            "rejected_input_ids": self.rejected_input_ids,
            "rejected_prompt_length": self.rejected_prompt_length,
        }


def tokenise_dpo_example(
    example: dict,
    tokenizer,
    *,
    max_length: int = 1024,
    template=DPO_DEFAULT_TEMPLATE,
) -> TokenisedDPOExample | None:
    """Render → tokenise both halves → return None if prompt alone too long
    or either response degenerates to zero tokens after stripping."""
    prompt_text = template.render_prompt(example["prompt"], "")
    eos = tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(prompt_ids) >= max_length:
        return None

    def _tokenise_full(completion: str) -> list[int] | None:
        completion = completion.strip()
        if not completion:
            return None
        full_text = prompt_text + completion + eos
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prompt_ids):
            return None
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            # Force EOS on the truncated tail so the model still sees a
            # termination signal (mirrors the SFT EOS-after-truncation fix).
            if tokenizer.eos_token_id is not None:
                full_ids[-1] = tokenizer.eos_token_id
        return full_ids

    chosen_ids = _tokenise_full(example["chosen"])
    rejected_ids = _tokenise_full(example["rejected"])
    if chosen_ids is None or rejected_ids is None:
        return None

    return TokenisedDPOExample(
        chosen_input_ids=chosen_ids,
        chosen_prompt_length=len(prompt_ids),
        rejected_input_ids=rejected_ids,
        rejected_prompt_length=len(prompt_ids),
    )


# ── Torch Dataset ────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    """In-memory list of TokenisedDPOExample dicts."""

    def __init__(self, examples: list[TokenisedDPOExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx].to_dict()


# ── Top-level entry point ────────────────────────────────────────────────────

@dataclass
class DPODatasetBundle:
    train: DPODataset
    val: DPODataset
    stats: dict


def build_dpo_datasets(
    *,
    hf_dataset_name: str = "Intel/orca_dpo_pairs",
    hf_split: str = "train",
    tokenizer=None,
    template=DPO_DEFAULT_TEMPLATE,
    max_length: int = 1024,
    val_fraction: float = 0.05,
    seed: int = 42,
    decontam_texts: Iterable[str] | None = None,
    max_examples: int | None = None,
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
) -> DPODatasetBundle:
    """Load + decontaminate + tokenise + split a DPO preference dataset.

    Decontamination uses the SHA-1 fingerprint of `(prompt + chosen)` AND
    `(prompt + rejected)` against the benchmark index — drop the pair if
    EITHER side overlaps a benchmark string.
    """
    if tokenizer is None:
        raise ValueError("DPO dataset building requires an initialised tokenizer.")

    from datasets import load_dataset  # type: ignore[import]

    log.info("Loading HF preference dataset: %s (split=%s)", hf_dataset_name, hf_split)
    raw = load_dataset(
        hf_dataset_name, split=hf_split,
        cache_dir=hf_cache_dir, token=hf_token,
    )
    raw_n = len(raw)
    log.info("Loaded %d raw preference rows.", raw_n)

    normalised: list[dict] = []
    dropped_schema = 0
    for row in raw:
        try:
            normalised.append(normalise_dpo_example(dict(row)))
        except ValueError:
            dropped_schema += 1
    log.info("Schema-normalised %d rows (dropped %d malformed).",
             len(normalised), dropped_schema)

    benchmark_index = build_decontam_index(decontam_texts or [])
    decontaminated: list[dict] = []
    dropped_contam = 0
    for ex in normalised:
        bag_chosen = [{"instruction": ex["prompt"], "response": ex["chosen"]}]
        bag_rejected = [{"instruction": ex["prompt"], "response": ex["rejected"]}]
        kept_c, dropped_c = filter_contaminated(bag_chosen, benchmark_index)
        kept_r, dropped_r = filter_contaminated(bag_rejected, benchmark_index)
        if dropped_c or dropped_r:
            dropped_contam += 1
            continue
        decontaminated.append(ex)
    log.info("Decontaminated against %d benchmark hashes (dropped %d pairs).",
             len(benchmark_index), dropped_contam)

    if max_examples is not None:
        decontaminated = decontaminated[: int(max_examples)]
        log.info("Capped to first %d examples.", len(decontaminated))

    tokenised: list[TokenisedDPOExample] = []
    dropped_tok = 0
    for ex in decontaminated:
        tok = tokenise_dpo_example(ex, tokenizer, max_length=max_length, template=template)
        if tok is None:
            dropped_tok += 1
            continue
        tokenised.append(tok)
    log.info("Tokenised %d preference pairs (dropped %d).",
             len(tokenised), dropped_tok)

    import random
    rng = random.Random(int(seed))
    indices = list(range(len(tokenised)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(tokenised) * float(val_fraction))))
    val_idx = set(indices[:n_val])
    train_examples = [tokenised[i] for i in range(len(tokenised)) if i not in val_idx]
    val_examples = [tokenised[i] for i in range(len(tokenised)) if i in val_idx]

    stats = {
        "raw": raw_n,
        "dropped_schema": dropped_schema,
        "dropped_contaminated": dropped_contam,
        "dropped_tokenisation": dropped_tok,
        "kept": len(tokenised),
        "train": len(train_examples),
        "val": len(val_examples),
    }
    log.info("DPO dataset stats: %s", stats)

    return DPODatasetBundle(
        train=DPODataset(train_examples),
        val=DPODataset(val_examples),
        stats=stats,
    )
