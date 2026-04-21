"""SFT data pipeline: load, decontaminate, tokenise, split.

This module converts a Hugging Face instruction dataset (default: Alpaca
per VL07 slide 48's explicit recommendation) into torch Datasets that the
trainer can consume. The five logical steps correspond to VL07 §2
(Data) and §3 (Warning: Catastrophic Forgetting):

1. Load from HF Hub (``datasets.load_dataset``). Cyril's SFT guide
   recommended `trl-lib/Capybara` or `HuggingFaceH4/ultrachat_200k`; the
   course recommendation in VL07 slide 48 is Alpaca. We default to Alpaca
   because it matches the Alpaca template we committed to in §4 of the
   SFT.md plan.

2. Normalise to internal schema ``{instruction, input, response}``. See
   ``template.normalise_hf_example``.

3. Decontaminate against the four public leaderboard test splits
   (LAMBADA, HellaSwag, WinoGrande, OpenBookQA). The phase-1 SHA-1
   machinery from ``src/data/preprocess.py`` is reused here. VL07 does not
   mention decontamination for SFT explicitly, but the fact sheet
   (§4.3) says "You need to make sure that the Test Datasets are not part
   of the training in any form" — which applies transitively to every
   training stage, including SFT.

4. Render with the Alpaca template (``template.render_example``), tokenise
   with the repo's GPT-2+pad tokenizer (``src.utils.build_tokenizer``), and
   record ``prompt_length`` so the collator can mask instruction tokens.

5. Split into (train, val). We keep 5 % for validation. Validation is used
   both for early stopping and for the VL07 slide 25 catastrophic-
   forgetting tripwire — when SFT-val loss is still dropping but
   Wikitext-103 PPL rises materially, we are overtraining.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import Dataset

from src.post_training.sft.template import (
    AlpacaTemplate,
    DEFAULT_ALPACA_TEMPLATE,
    normalise_hf_example,
    render_example,
)


log = logging.getLogger("parrotllm.sft.data")


# ── Decontamination (mirrors src/data/preprocess.py phase-1 SHA-1 hashing) ───

def _normalise_for_hash(text: str) -> str:
    """Lowercase + whitespace-collapse, matching preprocess.py's fingerprint rule."""
    return " ".join(text.lower().split())


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_decontam_index(texts: Iterable[str]) -> set[str]:
    """Return a set of SHA-1 fingerprints for all benchmark test examples.

    Mirrors the structure of ``src/data/preprocess.py`` phase 1 so the SFT
    pipeline's decontamination is byte-compatible with pretraining's:
    the same string that was dropped from pretraining will be dropped here.
    """
    index: set[str] = set()
    for t in texts:
        if not t:
            continue
        index.add(_sha1(_normalise_for_hash(t)))
    return index


def filter_contaminated(
    examples: list[dict],
    benchmark_index: set[str],
) -> tuple[list[dict], int]:
    """Drop examples whose (instruction + response) concat hash is in the index.

    Returns ``(kept, num_dropped)``.
    """
    if not benchmark_index:
        return examples, 0
    kept: list[dict] = []
    dropped = 0
    for ex in examples:
        blob = f"{ex.get('instruction','')} {ex.get('response','')}"
        if _sha1(_normalise_for_hash(blob)) in benchmark_index:
            dropped += 1
            continue
        kept.append(ex)
    return kept, dropped


# ── Tokenisation ─────────────────────────────────────────────────────────────

@dataclass
class TokenisedExample:
    """One fully-tokenised SFT example.

    ``input_ids`` is the full (prompt + response + eos) id list.
    ``prompt_length`` is the number of tokens in the prompt; the collator
    uses it to set ``labels[:prompt_length] = -100`` (VL07 slide 15).
    """

    input_ids: list[int]
    prompt_length: int

    def to_dict(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "prompt_length": self.prompt_length,
        }


def tokenise_example(
    example: dict,
    tokenizer,
    template: AlpacaTemplate = DEFAULT_ALPACA_TEMPLATE,
    *,
    max_length: int = 1024,
    append_eos: bool = True,
) -> TokenisedExample | None:
    """Render → tokenise → compute mask boundary for one example.

    Returns ``None`` if the example is too short to be useful (e.g. empty
    response after stripping) or if the prompt alone already exceeds
    ``max_length`` (no room for a response to learn from).
    """
    eos_token = tokenizer.eos_token if append_eos else ""
    try:
        prompt, full_text = render_example(example, template=template, eos_token=eos_token)
    except ValueError:
        return None

    # Tokenise the prompt and the full sequence separately. We tokenise the
    # prompt ONCE to find the boundary; the full sequence tokenisation is
    # what actually goes to the model. Using the prompt's token count as
    # the boundary assumes tokenisation of `prompt + response` is the
    # concatenation of the two tokenisations — true for GPT-2 BPE with
    # `add_special_tokens=False` because we control the delimiters (plain
    # text `### Response:\n`). This is the single place where an Alpaca
    # format is strictly simpler than ChatML.
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    if len(full_ids) <= len(prompt_ids):
        # Response collapsed to zero tokens (e.g. whitespace-only).
        return None

    # Truncate from the right (drop tail of response), never the prompt.
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    if len(prompt_ids) >= max_length:
        # No room left to learn a response. Skip rather than emit a 0-loss
        # example that silently dilutes the batch gradient.
        return None

    return TokenisedExample(
        input_ids=full_ids,
        prompt_length=len(prompt_ids),
    )


# ── Torch Dataset ────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """In-memory list-of-dicts dataset.

    At the scale we care about (Alpaca ≈ 52k rows × ≤1024 tokens ≈ 50M
    token ids = ~400MB uint16), holding the tokenised corpus in RAM is
    trivial and saves a lot of collator complexity vs. streaming.
    """

    def __init__(self, examples: list[TokenisedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx].to_dict()


# ── Top-level entry point ────────────────────────────────────────────────────

@dataclass
class SFTDatasetBundle:
    train: SFTDataset
    val: SFTDataset
    stats: dict


def build_sft_datasets(
    *,
    hf_dataset_name: str = "tatsu-lab/alpaca",
    hf_split: str = "train",
    tokenizer=None,
    template: AlpacaTemplate = DEFAULT_ALPACA_TEMPLATE,
    max_length: int = 1024,
    val_fraction: float = 0.05,
    seed: int = 42,
    decontam_texts: Iterable[str] | None = None,
    max_examples: int | None = None,
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
) -> SFTDatasetBundle:
    """Load + decontaminate + tokenise + split an SFT dataset from HF Hub.

    Args:
        hf_dataset_name: HF dataset identifier. Default is Stanford Alpaca
            per VL07 slide 48's explicit recommendation for PikoGPT.
        hf_split: split name to pull (Alpaca has only "train").
        tokenizer: a GPT-2 tokenizer with `<|pad|>` added. Usually
            ``src.utils.build_tokenizer()``.
        template: the Alpaca chat template. Default is the frozen copy
            defined in ``template.py``.
        max_length: hard truncation ceiling. Must be ≤ model context
            length (1024 per PikoGPT fact sheet).
        val_fraction: held-out validation fraction. 5% is standard; keep
            it small because Alpaca is already small.
        seed: deterministic split seed. 42 matches the repo-wide default
            set in ``main.py``.
        decontam_texts: an iterable of benchmark test-set strings. If
            provided, examples whose normalised hash matches are dropped
            (VL07 §2 "quality beats quantity" + fact sheet §4.3).
        max_examples: optional cap for smoke testing.
        hf_cache_dir / hf_token: passed through to ``datasets.load_dataset``.

    Returns:
        ``SFTDatasetBundle(train, val, stats)`` where ``stats`` is a dict
        of counts used in logging / tech-report tables.
    """
    if tokenizer is None:
        raise ValueError("SFT dataset building requires an initialised tokenizer.")

    # Lazy import so the repo still imports cleanly when `datasets` is
    # not yet installed (e.g. on a fresh clone before `uv sync`).
    from datasets import load_dataset  # type: ignore[import]

    log.info("Loading HF dataset: %s (split=%s)", hf_dataset_name, hf_split)
    raw = load_dataset(
        hf_dataset_name,
        split=hf_split,
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    raw_n = len(raw)
    log.info("Loaded %d raw examples.", raw_n)

    # 2. Normalise to internal schema. Each raw row → {instruction, input, response}.
    normalised: list[dict] = []
    dropped_schema = 0
    for row in raw:
        try:
            normalised.append(normalise_hf_example(dict(row)))
        except ValueError:
            dropped_schema += 1
    log.info(
        "Schema-normalised %d rows (dropped %d malformed).",
        len(normalised), dropped_schema,
    )

    # 3. Decontamination (VL07 §2 + fact sheet §4.3).
    benchmark_index = build_decontam_index(decontam_texts or [])
    normalised, num_contaminated = filter_contaminated(normalised, benchmark_index)
    log.info(
        "Decontaminated against %d benchmark hashes (dropped %d rows).",
        len(benchmark_index), num_contaminated,
    )

    if max_examples is not None:
        normalised = normalised[: int(max_examples)]
        log.info("Capped to first %d examples (smoke-test mode).", len(normalised))

    # 4. Render + tokenise. Per-example failures are skipped silently (with a
    # running count) rather than crashing the whole job — the Alpaca corpus
    # has a handful of degenerate rows with empty responses.
    tokenised: list[TokenisedExample] = []
    dropped_empty = 0
    dropped_too_long = 0
    for ex in normalised:
        tok = tokenise_example(
            ex, tokenizer, template=template,
            max_length=max_length, append_eos=True,
        )
        if tok is None:
            # Distinguish the two failure modes for logging clarity.
            if ex.get("response") and len(ex["response"].strip()) > 0:
                dropped_too_long += 1
            else:
                dropped_empty += 1
            continue
        tokenised.append(tok)
    log.info(
        "Tokenised %d examples (dropped %d empty, %d prompt-too-long).",
        len(tokenised), dropped_empty, dropped_too_long,
    )

    # 5. Split. Deterministic shuffle so a re-run with the same seed gives
    # exactly the same val set — essential for apples-to-apples comparison
    # between ablations.
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
        "dropped_contaminated": num_contaminated,
        "dropped_empty": dropped_empty,
        "dropped_too_long": dropped_too_long,
        "kept": len(tokenised),
        "train": len(train_examples),
        "val": len(val_examples),
    }
    log.info("SFT dataset stats: %s", stats)

    return SFTDatasetBundle(
        train=SFTDataset(train_examples),
        val=SFTDataset(val_examples),
        stats=stats,
    )
