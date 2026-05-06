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
    DEFAULT_RAW_TEMPLATE,
    RawCompletionTemplate,
    normalise_hf_example,
    render_example,
)
from src.post_training.hf_cache import cleanup_hf_dataset_cache


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


# ── Benchmark loaders for decontam (SFT.md §3.3 "Mandatory") ────────────────

def _load_lambada() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    for row in ds:
        yield row["text"]


def _load_hellaswag() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    # Validation is the standard dev/test split for HellaSwag (test is hidden).
    ds = load_dataset("Rowan/hellaswag", split="validation")
    for row in ds:
        yield f"{row['ctx_a']} {row['ctx_b']}"
        for ending in row["endings"]:
            yield ending


def _load_winogrande() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset(
        "allenai/winogrande", "winogrande_xl",
        split="validation", trust_remote_code=True,
    )
    for row in ds:
        yield row["sentence"]


def _load_openbookqa() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    for row in ds:
        yield row["question_stem"]
        for choice in row["choices"]["text"]:
            yield choice


# ── Extended hidden-bench-safe decontam set ─────────────────────────────
# Synthetic raw-format mixin draws from public Q&A train splits (SciQ-train,
# ARC-Easy-train, CommonsenseQA-train, PIQA-train, MMLU-train). Hidden tests
# in the course leaderboard most likely come from the *test* splits of these
# same families. Hash-decontaminating against the test/validation splits
# below is a belt-and-suspenders defence: even if the sourcing pipeline
# accidentally pulls a test row, we drop it here.

def _load_arc_easy() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for row in ds:
        yield row["question"]
        for choice in row["choices"]["text"]:
            yield choice


def _load_arc_challenge() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for row in ds:
        yield row["question"]
        for choice in row["choices"]["text"]:
            yield choice


def _load_boolq() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("google/boolq", split="validation")
    for row in ds:
        yield row["question"]
        yield row["passage"]


def _load_commonsenseqa() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("tau/commonsense_qa", split="validation")
    for row in ds:
        yield row["question"]
        for choice in row["choices"]["text"]:
            yield choice


def _load_sciq() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("allenai/sciq", split="test")
    for row in ds:
        yield row["question"]
        yield row["correct_answer"]
        for key in ("distractor1", "distractor2", "distractor3"):
            yield row.get(key, "")


def _load_mmlu() -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset("cais/mmlu", "all", split="test")
    for row in ds:
        yield row["question"]
        for choice in row["choices"]:
            yield choice


# Registry of canonical leaderboard benchmarks (PikoGPT fact sheet §4.3)
# plus the hidden-bench-safe extension splits.
# Keys are the short names users put in `sft.decontam_benchmarks`.
DECONTAM_LOADERS: dict[str, Callable[[], Iterable[str]]] = {
    "lambada": _load_lambada,
    "hellaswag": _load_hellaswag,
    "winogrande": _load_winogrande,
    "openbookqa": _load_openbookqa,
    # Extended: hash-decontam against test/validation splits of public Q&A
    # families that v6's synthetic mixin draws from on the train side.
    "arc_easy": _load_arc_easy,
    "arc_challenge": _load_arc_challenge,
    "boolq": _load_boolq,
    "commonsenseqa": _load_commonsenseqa,
    "sciq": _load_sciq,
    "mmlu": _load_mmlu,
}


def load_decontam_texts(
    names: Iterable[str],
    *,
    cleanup_hf_cache: bool = False,
) -> Iterable[str]:
    """Yield benchmark texts for the configured names.

    Resolves each short name in `names` to a registered loader (see
    `DECONTAM_LOADERS`) and streams its texts. Unknown names raise
    immediately so a YAML typo cannot silently disable decontamination —
    invalid leaderboard scores are the worst kind of failure here.

    Caller is `run_sft`, which collects the result into a list and passes
    it to `build_sft_datasets(decontam_texts=...)`.
    """
    for name in names:
        loader = DECONTAM_LOADERS.get(name)
        if loader is None:
            known = ", ".join(sorted(DECONTAM_LOADERS))
            raise ValueError(
                f"Decontam: unknown benchmark '{name}'. Known: {known}."
            )
        log.info("Decontam: loading benchmark '%s'", name)
        try:
            yield from loader()
        finally:
            if cleanup_hf_cache:
                cleanup_hf_dataset_cache()


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
    template=DEFAULT_ALPACA_TEMPLATE,
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
    # Force the final token back to EOS so the model still learns to
    # terminate on over-length examples — without this, generation at
    # inference runs to max_tokens with garbage on any prompt resembling
    # a truncated training example (SFT.md §9 risk row).
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        if append_eos and tokenizer.eos_token_id is not None:
            full_ids[-1] = tokenizer.eos_token_id

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
    template=DEFAULT_ALPACA_TEMPLATE,
    max_length: int = 1024,
    val_fraction: float = 0.05,
    seed: int = 42,
    decontam_texts: Iterable[str] | None = None,
    max_examples: int | None = None,
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
    cleanup_hf_cache: bool = False,
    synthetic_jsonl_path: str | None = None,
    synthetic_oversample: int = 1,
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
        cleanup_hf_cache: remove HF dataset cache files after rows have been
            materialized/tokenized.

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
        "Tokenised %d Alpaca examples (dropped %d empty, %d prompt-too-long).",
        len(tokenised), dropped_empty, dropped_too_long,
    )
    raw = None
    if cleanup_hf_cache:
        cleanup_hf_dataset_cache(cache_dir=hf_cache_dir)

    # 4b. Optional synthetic raw-format mixin.
    #
    # The synthetic JSONL holds {instruction, response} pairs whose
    # `instruction` text is the *exact* prompt the leaderboard runner
    # will send (e.g. ``"Context: ...\nA) ...\nB) ...\nAnswer:"``) and
    # whose `response` is the bare completion (e.g. ``" B"``). These are
    # rendered with RawCompletionTemplate so the model trains on the
    # byte-identical string the runner will pass at inference.
    #
    # synthetic_oversample lets us hit a target per-batch share without
    # touching the trainer: e.g. with 50k Alpaca rows and 2.5k synthetic,
    # oversample=5 gives 12.5k synthetic rows = ~20% of the SFT pool.
    synthetic_kept = 0
    synthetic_dropped_contaminated = 0
    synthetic_dropped_empty = 0
    synthetic_dropped_too_long = 0
    if synthetic_jsonl_path:
        log.info("Loading synthetic raw-format JSONL: %s", synthetic_jsonl_path)
        syn_raw = load_dataset(
            "json",
            data_files=synthetic_jsonl_path,
            split="train",
            cache_dir=hf_cache_dir,
        )
        log.info("Loaded %d synthetic raw rows.", len(syn_raw))

        syn_normalised: list[dict] = []
        for row in syn_raw:
            try:
                syn_normalised.append(normalise_hf_example(dict(row)))
            except ValueError:
                continue
        # Same hash-decontam as the Alpaca side. Synthetic should never
        # contain leaderboard test items by construction (programmatic
        # templates) or by upstream filter (public Q&A reformatting), but
        # we run the check anyway as belt-and-suspenders.
        syn_normalised, num_contam = filter_contaminated(syn_normalised, benchmark_index)
        synthetic_dropped_contaminated = num_contam
        log.info(
            "Synthetic decontamination: dropped %d (kept %d).",
            num_contam, len(syn_normalised),
        )

        syn_tokenised: list[TokenisedExample] = []
        for ex in syn_normalised:
            tok = tokenise_example(
                ex, tokenizer, template=DEFAULT_RAW_TEMPLATE,
                max_length=max_length, append_eos=True,
            )
            if tok is None:
                if ex.get("response") and len(ex["response"].strip()) > 0:
                    synthetic_dropped_too_long += 1
                else:
                    synthetic_dropped_empty += 1
                continue
            syn_tokenised.append(tok)
        log.info(
            "Tokenised %d synthetic examples (dropped %d empty, %d too-long).",
            len(syn_tokenised), synthetic_dropped_empty, synthetic_dropped_too_long,
        )
        syn_raw = None
        if cleanup_hf_cache:
            cleanup_hf_dataset_cache(cache_dir=hf_cache_dir)

        if synthetic_oversample > 1:
            syn_tokenised = syn_tokenised * int(synthetic_oversample)
            log.info(
                "Oversampled synthetic %dx -> %d rows.",
                int(synthetic_oversample), len(syn_tokenised),
            )

        synthetic_kept = len(syn_tokenised)
        tokenised = tokenised + syn_tokenised

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
        "synthetic_kept": synthetic_kept,
        "synthetic_dropped_contaminated": synthetic_dropped_contaminated,
        "synthetic_dropped_empty": synthetic_dropped_empty,
        "synthetic_dropped_too_long": synthetic_dropped_too_long,
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
