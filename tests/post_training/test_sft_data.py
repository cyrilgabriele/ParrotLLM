"""Tests for the SFT data pipeline (load → decontaminate → tokenise → split).

Covers correctness invariants that have observable downstream consequences:

- Right-truncation must preserve the EOS token. Without this, the model
  never learns to stop on long examples (SFT.md §9 "Wrong EOS / generation
  never stops" risk row).
- Decontamination must drop examples whose normalised hash matches a
  benchmark fingerprint (SFT.md §3.3 "Mandatory" — leaked test rows
  invalidate leaderboard scores).
"""

from __future__ import annotations

import pytest

from src.post_training.sft.data import (
    build_decontam_index,
    filter_contaminated,
    tokenise_example,
)
from src.utils import build_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


# ── EOS preservation across right-truncation ─────────────────────────────────

def test_tokenise_example_preserves_eos_after_right_truncation(tokenizer):
    """When full_ids exceeds max_length, the truncated sequence must still
    end in EOS. Otherwise the model never learns to terminate on long
    examples and inference runs to max_tokens with garbage."""
    eos_id = tokenizer.eos_token_id
    # Short instruction + long response: the Alpaca prompt prefix is ~30 tokens,
    # so max_length=64 leaves room for some response that gets truncated.
    long_response = "word " * 400
    example = {
        "instruction": "Hi.",
        "input": "",
        "response": long_response.strip(),
    }
    tok = tokenise_example(example, tokenizer, max_length=64, append_eos=True)
    assert tok is not None, "Sanity: tokeniser returned a result for a long example."
    assert len(tok.input_ids) == 64, "max_length truncation enforced."
    assert tok.input_ids[-1] == eos_id, (
        "Right-truncation dropped the EOS token. Model will never learn to "
        "terminate on over-length examples — generation runs to max_tokens."
    )


def test_tokenise_example_no_truncation_keeps_eos(tokenizer):
    """Sanity: when no truncation is needed, EOS is the last token."""
    eos_id = tokenizer.eos_token_id
    example = {"instruction": "Say hi.", "input": "", "response": "Hi."}
    tok = tokenise_example(example, tokenizer, max_length=1024, append_eos=True)
    assert tok is not None
    assert tok.input_ids[-1] == eos_id


def test_tokenise_example_skips_when_prompt_alone_exceeds_max_length(tokenizer):
    """If the prompt alone is too long there is no room for a response —
    return None rather than emitting a 0-loss example."""
    huge_instruction = "explain this in detail " * 200  # >>32 tokens
    example = {
        "instruction": huge_instruction.strip(),
        "input": "",
        "response": "Short answer.",
    }
    tok = tokenise_example(example, tokenizer, max_length=16, append_eos=True)
    assert tok is None


# ── Decontamination ──────────────────────────────────────────────────────────

def test_filter_contaminated_drops_exact_match():
    benchmark = ["The quick brown fox jumps over the lazy dog."]
    index = build_decontam_index(benchmark)
    examples = [
        # Exact match (normalised) — should be dropped.
        {"instruction": "The quick brown fox", "response": "jumps over the lazy dog."},
        # Unrelated — should be kept.
        {"instruction": "What is 2 + 2?", "response": "4"},
    ]
    kept, dropped = filter_contaminated(examples, index)
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["instruction"] == "What is 2 + 2?"


def test_filter_contaminated_normalises_whitespace_and_case():
    benchmark = ["Hello   World"]
    index = build_decontam_index(benchmark)
    examples = [{"instruction": "hello world", "response": ""}]
    _, dropped = filter_contaminated(examples, index)
    assert dropped == 1, "Decontam should match across case / whitespace differences."


def test_filter_contaminated_with_empty_index_is_noop():
    examples = [{"instruction": "Q", "response": "A"}]
    kept, dropped = filter_contaminated(examples, set())
    assert dropped == 0
    assert kept == examples


# ── Benchmark text loading (decontam pipeline integration) ──────────────────

def test_load_decontam_texts_rejects_unknown_benchmark():
    """Unknown benchmark names must raise immediately so a typo in YAML
    doesn't silently disable decontamination."""
    from src.post_training.sft.data import load_decontam_texts
    import pytest

    with pytest.raises(ValueError, match="unknown benchmark"):
        list(load_decontam_texts(["not-a-real-benchmark"]))


def test_load_decontam_texts_empty_list_is_noop():
    from src.post_training.sft.data import load_decontam_texts

    assert list(load_decontam_texts([])) == []


def test_load_decontam_texts_uses_registered_loader(monkeypatch):
    """The loader for a known benchmark must be invoked and its texts
    returned. We monkeypatch the registry so the test doesn't hit HF."""
    from src.post_training.sft import data as data_mod

    def fake_loader():
        return ["alpha", "beta", "gamma"]

    monkeypatch.setitem(data_mod.DECONTAM_LOADERS, "fake-bench", fake_loader)
    out = list(data_mod.load_decontam_texts(["fake-bench"]))
    assert out == ["alpha", "beta", "gamma"]
