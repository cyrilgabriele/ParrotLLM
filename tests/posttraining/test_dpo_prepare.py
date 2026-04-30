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
