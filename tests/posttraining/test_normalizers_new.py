"""Tests for newly-added MC normalizers (CosmosQA, SocialIQa, etc.)."""

from __future__ import annotations

import re

import pytest

from configs import SFTSourceConfig
from src.posttraining.prepare import _normalize_cosmos_qa_record

_CHOICE_RE = re.compile(r"^([A-Z])\) (.+?)$", re.MULTILINE)


def _parse_choices(prompt: str) -> dict[str, str]:
    """Parse 'A) ... B) ...' lines from an MC prompt into a {letter: choice_text} map."""
    return dict(_CHOICE_RE.findall(prompt))


def _src(loader: str) -> SFTSourceConfig:
    return SFTSourceConfig(
        name=f"{loader}_test",
        loader=loader,
        path=f"allenai/{loader}",
        split="train",
        target_examples=1,
    )


def test_cosmos_qa_basic_record():
    rec = {
        "id": "cosmos_001",
        "context": "We were driving to the cabin when the storm hit.",
        "question": "What is most likely true about the trip?",
        "answer0": "The trip was uneventful and quick.",
        "answer1": "The driver pulled over due to weather.",
        "answer2": "They reached the cabin in record time.",
        "answer3": "They turned around and went home.",
        "label": 1,
    }

    result = _normalize_cosmos_qa_record(rec, _src("cosmos_qa"))
    assert result is not None
    messages, meta = result

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    user_prompt = messages[0]["content"]
    assert "Context:" in user_prompt or "Passage:" in user_prompt or rec["context"] in user_prompt
    assert rec["question"] in user_prompt
    # All four answers must appear in the prompt regardless of permuted order.
    for ans in (rec["answer0"], rec["answer1"], rec["answer2"], rec["answer3"]):
        assert ans in user_prompt
    assert user_prompt.endswith("Answer:")

    gold_letter = messages[1]["content"]
    assert gold_letter in {"A", "B", "C", "D"}

    # Verify the gold letter actually maps back to the original gold answer text.
    label_to_answer = _parse_choices(user_prompt)
    assert label_to_answer[gold_letter] == rec["answer1"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("label", "not_an_int"),  # non-int label
        ("label", 5),               # out-of-range label (>3)
        ("label", -1),              # out-of-range label (<0)
        ("label", None),            # missing label
        ("context", ""),            # empty context
        ("question", ""),           # empty question
        ("answer0", ""),            # empty answer
    ],
)
def test_cosmos_qa_rejects_invalid(field: str, value):
    rec = {
        "id": "cosmos_002",
        "context": "ctx",
        "question": "q",
        "answer0": "a",
        "answer1": "b",
        "answer2": "c",
        "answer3": "d",
        "label": 0,
    }
    rec[field] = value
    assert _normalize_cosmos_qa_record(rec, _src("cosmos_qa")) is None
