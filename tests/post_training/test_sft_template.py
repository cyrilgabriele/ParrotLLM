"""Tests for the Alpaca SFT template and HF schema normalisation.

Invariants enforced here (all drawn from VL07):

- Slide 32 "Critical rule": the rendered prompt string is deterministic so
  training-time and inference-time templates can be compared byte-for-byte.
- Slide 32 "Alpaca — simplest choice": no new special tokens introduced —
  the template is plain text.
- Slide 17 schema tolerance: normalise the three common HF instruction
  schemas ({instruction, output}, {prompt, completion}, {messages[]}) to
  our single internal shape.
"""

from __future__ import annotations

import pytest

from src.post_training.sft.template import (
    DEFAULT_ALPACA_TEMPLATE,
    normalise_hf_example,
    render_example,
)


def test_render_prompt_without_input_contains_response_marker():
    prompt = DEFAULT_ALPACA_TEMPLATE.render_prompt("Explain gravity to a child.")
    assert "### Instruction:" in prompt
    assert "### Response:\n" in prompt
    # The last characters of the prompt must be the response marker. This is
    # what the collator uses as the mask boundary — any trailing whitespace
    # here would shift the token ids and silently break masking.
    assert prompt.endswith("### Response:\n")


def test_render_prompt_with_input_includes_input_block():
    prompt = DEFAULT_ALPACA_TEMPLATE.render_prompt(
        "Translate to German.",
        input_text="I love transformers.",
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:\n" in prompt


def test_render_full_appends_response_and_eos():
    prompt, full = render_example(
        {"instruction": "Say hi.", "input": "", "response": "Hello."},
        eos_token="<|eos|>",
    )
    assert full.startswith(prompt)
    assert full.endswith("Hello." + "<|eos|>")


def test_render_full_is_byte_deterministic():
    """The same inputs must produce byte-identical output across calls —
    otherwise the VL07 slide 32 "critical rule" is violated non-obviously."""
    ex = {"instruction": "A", "input": "", "response": "B"}
    assert render_example(ex) == render_example(ex)


def test_normalise_alpaca_schema():
    raw = {"instruction": "Q", "input": "ctx", "output": "A"}
    n = normalise_hf_example(raw)
    assert n == {"instruction": "Q", "input": "ctx", "response": "A"}


def test_normalise_prompt_completion_schema():
    raw = {"prompt": "Q", "completion": "A"}
    n = normalise_hf_example(raw)
    assert n == {"instruction": "Q", "input": "", "response": "A"}


def test_normalise_messages_schema_single_turn():
    raw = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]}
    n = normalise_hf_example(raw)
    assert n["instruction"] == "Q"
    assert n["response"] == "A"


def test_normalise_messages_schema_with_system_prompt():
    raw = {"messages": [
        {"role": "system", "content": "You are ParrotLLM."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
    ]}
    n = normalise_hf_example(raw)
    assert "You are ParrotLLM." in n["instruction"]
    assert "Hi." in n["instruction"]
    assert n["response"] == "Hello!"


def test_normalise_unknown_schema_raises():
    with pytest.raises(ValueError):
        normalise_hf_example({"foo": "bar"})


def test_render_example_rejects_empty_response():
    with pytest.raises(ValueError):
        render_example({"instruction": "Q", "input": "", "response": ""})
