"""DPO prepare emits packed JSONL with the contract used by the trainer."""
import json
from pathlib import Path

import pytest

from src.posttraining.dpo.prepare import (
    PreparedPreferencePair,
    parse_hh_rlhf_record,
    pack_pair,
)


def test_parse_hh_rlhf_extracts_prompt_chosen_rejected() -> None:
    raw = {
        "chosen": "\n\nHuman: What is 2+2?\n\nAssistant: 4.",
        "rejected": "\n\nHuman: What is 2+2?\n\nAssistant: idk lol",
    }
    parsed = parse_hh_rlhf_record(raw)
    assert parsed is not None
    assert parsed.prompt.strip() == "What is 2+2?"
    assert parsed.chosen.strip() == "4."
    assert parsed.rejected.strip() == "idk lol"


def test_parse_hh_rlhf_returns_none_when_chosen_equals_rejected() -> None:
    raw = {
        "chosen": "\n\nHuman: hi\n\nAssistant: hi",
        "rejected": "\n\nHuman: hi\n\nAssistant: hi",
    }
    assert parse_hh_rlhf_record(raw) is None


def test_pack_pair_writes_shared_prompt_and_distinct_responses() -> None:
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    pair = PreparedPreferencePair(
        prompt="What is 2+2?",
        chosen="4.",
        rejected="idk lol",
    )
    packed = pack_pair(pair, tokenizer=tok, system_prompt="You are ParrotLLM.", max_seq_length=128)
    assert "prompt_tokens" in packed
    assert "chosen_tokens" in packed
    assert "rejected_tokens" in packed
    assert packed["prompt_len"] == len(packed["prompt_tokens"])
    # Both chosen and rejected sequences must start with the same prompt tokens.
    assert packed["chosen_tokens"][:packed["prompt_len"]] == packed["prompt_tokens"]
    assert packed["rejected_tokens"][:packed["prompt_len"]] == packed["prompt_tokens"]
    assert packed["chosen_tokens"] != packed["rejected_tokens"]


def test_pack_pair_renders_user_question_into_prompt() -> None:
    """Regression: prior bug rendered prompts as ONLY '\\n\\n### Response:\\n' (empty content).

    `render_conversation` requires `add_generation_prompt=True` to emit a single-user-message
    prompt; without it, it returns the empty string. The earlier code appended a manual
    response header to that empty string, producing a packed record where the user
    question was missing entirely. DPO would have trained on prompts containing zero
    question signal — silently.
    """
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    pair = PreparedPreferencePair(
        prompt="What is two plus two?",
        chosen="Four.",
        rejected="Seven.",
    )
    packed = pack_pair(
        pair,
        tokenizer=tok,
        system_prompt="You are ParrotLLM.",
        max_seq_length=128,
    )
    decoded_prompt = tok.decode(packed["prompt_tokens"])
    # User question text must be present.
    assert "two plus two" in decoded_prompt, (
        f"User question was lost during prompt rendering. "
        f"decoded prompt = {decoded_prompt!r}"
    )
    # System prompt must be present (Alpaca-merge style).
    assert "You are ParrotLLM" in decoded_prompt, (
        f"System prompt was lost. decoded prompt = {decoded_prompt!r}"
    )
    # Prompt should still end with the response header so the model knows where to start.
    assert decoded_prompt.rstrip().endswith("### Response:"), (
        f"Prompt should end with the assistant header. decoded prompt = {decoded_prompt!r}"
    )


def test_dpo_decontam_drops_leaked_pair() -> None:
    """A pair whose prompt matches a decontam-set entry must be dropped.

    Verifies the `_filter_decontaminated` helper supports both:
      - a flat `set[str]` of normalized eval-split prompt substrings, and
      - a `PromptContaminationIndex` (MinHash + 5-gram + Jaccard 0.8).
    """
    from src.posttraining.dpo.prepare import _filter_decontaminated
    from src.posttraining.prepare import PromptContaminationIndex

    pair_clean = {
        "prompt": "A clean MC question.\nA) one\nB) two\nAnswer:",
        "prompt_tokens": [1, 2, 3],
        "chosen_tokens": [1, 2, 3, 4],
        "rejected_tokens": [1, 2, 3, 5],
        "prompt_len": 3,
    }
    pair_leaked = {
        "prompt": "this is the leaked context that should be dropped\nA) x\nB) y\nAnswer:",
        "prompt_tokens": [1, 2, 3],
        "chosen_tokens": [1, 2, 3, 4],
        "rejected_tokens": [1, 2, 3, 5],
        "prompt_len": 3,
    }

    # Path 1: flat set of normalized substrings.
    decontam_set = {"this is the leaked context that should be dropped"}
    kept = _filter_decontaminated([pair_clean, pair_leaked], decontam_set)
    assert len(kept) == 1
    assert "leaked" not in kept[0]["prompt"].lower()

    # Path 2: PromptContaminationIndex (MinHash, used in production).
    # Add the exact normalized prompt so the index's exact-hash short-circuit
    # fires; the MinHash + Jaccard 0.8 path also covers near-duplicates but
    # needs >= 0.8 shingle overlap, so we use exact-match here for determinism.
    index = PromptContaminationIndex()
    index.add(pair_leaked["prompt"])
    kept_idx = _filter_decontaminated([pair_clean, pair_leaked], index)
    assert len(kept_idx) == 1
    assert "leaked" not in kept_idx[0]["prompt"].lower()


def test_build_continuation_dpo_pair_basic():
    from src.posttraining.dpo.prepare import _build_continuation_dpo_pair
    from src.utils import build_tokenizer
    import random

    tokenizer = build_tokenizer()
    rng = random.Random(0)

    pair = _build_continuation_dpo_pair(
        user_prompt="Context: A man stands on a roof. He",
        correct_continuation="starts pulling up roofing.",
        distractor_continuations=[
            "is using wrap to wrap a pair of skis.",
            "is holding a rubik's cube.",
            "starts pulling up tomatoes.",
        ],
        tokenizer=tokenizer,
        system_prompt="You are ParrotLLM.",
        max_seq_length=1024,
        rng=rng,
    )

    assert pair is not None
    assert "prompt_tokens" in pair
    assert "chosen_tokens" in pair
    assert "rejected_tokens" in pair
    assert pair["chosen_tokens"] != pair["rejected_tokens"]
    assert len(pair["chosen_tokens"]) > pair["prompt_len"]
    assert len(pair["rejected_tokens"]) > pair["prompt_len"]


def test_build_continuation_dpo_pair_rejects_no_distractors():
    from src.posttraining.dpo.prepare import _build_continuation_dpo_pair
    from src.utils import build_tokenizer
    import random

    pair = _build_continuation_dpo_pair(
        user_prompt="Q?",
        correct_continuation="answer",
        distractor_continuations=[],
        tokenizer=build_tokenizer(),
        system_prompt="You are ParrotLLM.",
        max_seq_length=1024,
        rng=random.Random(0),
    )
    assert pair is None
