"""Tests for newly-added MC normalizers (CosmosQA, SocialIQa, etc.)."""

from __future__ import annotations

from configs import SFTSourceConfig
from src.posttraining.prepare import _normalize_cosmos_qa_record


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
    label_to_answer = {
        "A": user_prompt.split("A) ")[1].split("\n")[0],
        "B": user_prompt.split("B) ")[1].split("\n")[0],
        "C": user_prompt.split("C) ")[1].split("\n")[0],
        "D": user_prompt.split("D) ")[1].split("\n")[0],
    }
    assert label_to_answer[gold_letter] == rec["answer1"]


def test_cosmos_qa_rejects_missing_field():
    rec = {"id": "x", "context": "ctx", "question": "q"}  # no answers
    assert _normalize_cosmos_qa_record(rec, _src("cosmos_qa")) is None
