"""Tests for the DPO data pipeline (schema normalization + tokenization)."""

from __future__ import annotations

import pytest

from src.post_training.dpo.data import normalise_dpo_example, tokenise_dpo_example
from src.utils import build_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


# ── Schema normalization ────────────────────────────────────────────────────

def test_normalise_orca_dpo_pairs_schema():
    raw = {"system": "You are helpful.", "question": "What is 2+2?",
           "chosen": "4", "rejected": "5"}
    n = normalise_dpo_example(raw)
    assert "You are helpful." in n["prompt"]
    assert "What is 2+2?" in n["prompt"]
    assert n["chosen"] == "4"
    assert n["rejected"] == "5"


def test_normalise_orca_without_system():
    raw = {"question": "Q", "chosen": "A", "rejected": "B"}
    n = normalise_dpo_example(raw)
    assert n == {"prompt": "Q", "chosen": "A", "rejected": "B"}


def test_normalise_generic_string_schema():
    raw = {"prompt": "Hi", "chosen": "Hello", "rejected": "Bye"}
    n = normalise_dpo_example(raw)
    assert n == {"prompt": "Hi", "chosen": "Hello", "rejected": "Bye"}


def test_normalise_ultrafeedback_message_list_schema():
    """Ultrafeedback stores chosen/rejected as [{role, content}] lists.
    We extract the last assistant turn."""
    raw = {
        "prompt": "Solve this.",
        "chosen": [
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "content": "The answer is 42."},
        ],
        "rejected": [
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "content": "I don't know."},
        ],
    }
    n = normalise_dpo_example(raw)
    assert n["prompt"] == "Solve this."
    assert n["chosen"] == "The answer is 42."
    assert n["rejected"] == "I don't know."


def test_normalise_unknown_schema_raises():
    with pytest.raises(ValueError, match="Unrecognised DPO schema"):
        normalise_dpo_example({"foo": "bar"})


# ── Tokenization ────────────────────────────────────────────────────────────

def test_tokenise_dpo_example_produces_aligned_prompt_lengths(tokenizer):
    """The same prompt is shared by chosen and rejected → both halves
    must have the same prompt_length, so the collator's masks line up."""
    ex = {"prompt": "Say hi.", "chosen": "Hello!", "rejected": "Hi there."}
    tok = tokenise_dpo_example(ex, tokenizer, max_length=128)
    assert tok is not None
    assert tok.chosen_prompt_length == tok.rejected_prompt_length


def test_tokenise_dpo_example_eos_appended_to_both_halves(tokenizer):
    eos = tokenizer.eos_token_id
    ex = {"prompt": "Say hi.", "chosen": "Hi!", "rejected": "Bye."}
    tok = tokenise_dpo_example(ex, tokenizer, max_length=128)
    assert tok is not None
    assert tok.chosen_input_ids[-1] == eos
    assert tok.rejected_input_ids[-1] == eos


def test_tokenise_dpo_example_skips_when_either_side_empty(tokenizer):
    ex = {"prompt": "Say hi.", "chosen": "Hi", "rejected": "  "}
    assert tokenise_dpo_example(ex, tokenizer, max_length=128) is None


def test_tokenise_dpo_example_truncates_with_eos_preserved(tokenizer):
    """Long response → truncated → must still end in EOS (mirrors the SFT
    EOS-after-truncation fix)."""
    eos = tokenizer.eos_token_id
    long_resp = "word " * 400
    ex = {"prompt": "Hi", "chosen": long_resp, "rejected": "Bye."}
    tok = tokenise_dpo_example(ex, tokenizer, max_length=64)
    assert tok is not None
    assert len(tok.chosen_input_ids) == 64
    assert tok.chosen_input_ids[-1] == eos
