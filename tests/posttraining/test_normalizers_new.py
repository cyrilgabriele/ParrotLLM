"""Tests for newly-added MC normalizers (CosmosQA, SocialIQa, etc.)."""

from __future__ import annotations

import re

import pytest

from configs import SFTSourceConfig
from src.posttraining.prepare import (
    _normalize_bookcorpus_lambada_record,
    _normalize_cbt_record,
    _normalize_cosmos_qa_record,
    _normalize_flan_mc_record,
    _normalize_social_iqa_record,
)

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


def test_social_iqa_basic_record():
    rec = {
        "context": "Alex helped his friend study for the exam.",
        "question": "How would Alex feel afterwards?",
        "answerA": "happy and proud",
        "answerB": "sad and rejected",
        "answerC": "angry",
        "label": "1",  # SocialIQa uses 1-based string labels
    }
    result = _normalize_social_iqa_record(rec, _src("social_iqa"))
    assert result is not None
    messages, _meta = result
    user_prompt = messages[0]["content"]
    gold_letter = messages[1]["content"]

    # 3-way MC -> letters are A, B, C only.
    assert gold_letter in {"A", "B", "C"}
    # Verify all 3 choices are present in the rendered prompt.
    label_to_answer = _parse_choices(user_prompt)
    assert set(label_to_answer.keys()) == {"A", "B", "C"}
    assert sorted(label_to_answer.values()) == sorted(
        [rec["answerA"], rec["answerB"], rec["answerC"]]
    )
    # The gold letter must map back to the original answerA (label "1" = index 0).
    assert label_to_answer[gold_letter] == rec["answerA"]
    assert user_prompt.endswith("Answer:")


@pytest.mark.parametrize(
    "field, value",
    [
        ("label", "0"),    # 1-based: "0" is invalid
        ("label", "4"),    # out of range (only A/B/C)
        ("label", "abc"),  # non-numeric
        ("label", None),
        ("context", ""),
        ("question", ""),
        ("answerA", ""),
    ],
)
def test_social_iqa_rejects_invalid(field: str, value):
    rec = {
        "context": "ctx",
        "question": "q",
        "answerA": "a",
        "answerB": "b",
        "answerC": "c",
        "label": "1",
    }
    rec[field] = value
    assert _normalize_social_iqa_record(rec, _src("social_iqa")) is None


def test_cbt_basic_record():
    rec = {
        "sentences": [
            "Once upon a time there was a small village.",
            "The villagers were preparing for winter.",
            "Snow had begun to fall lightly.",
        ],
        "question": "Then XXXXX returned to the house.",
        "answer": "Mary",
        "options": ["village", "snow", "winter", "Mary", "field"],
    }

    result = _normalize_cbt_record(rec, _src("cbt"))
    assert result is not None
    messages, _meta = result
    user = messages[0]["content"]
    assistant = messages[1]["content"]

    # The assistant target IS the cloze answer (lowercased per LAMBADA convention).
    assert assistant.strip() == "mary"
    # Prompt ends with a single trailing space, ready for greedy continuation.
    assert user.endswith(" ")
    assert not user.endswith("  ")
    # Everything after the blank in the question must NOT appear in the prompt —
    # only the head (before XXXXX) is included, so the model is genuinely
    # predicting the cloze, not parroting the rest of the question.
    assert "returned to the house" not in user
    assert "XXXXX" not in user
    # The context sentences are present.
    assert "Once upon a time" in user
    # The question's prefix word is present.
    assert "Then" in user


@pytest.mark.parametrize(
    "field, value",
    [
        ("sentences", []),               # no context sentences
        ("sentences", None),             # missing
        ("question", ""),                # empty question
        ("question", "no blank here."),  # no XXXXX marker
        ("answer", ""),                  # empty answer
    ],
)
def test_cbt_rejects_invalid(field, value):
    rec = {
        "sentences": ["Sentence one.", "Sentence two."],
        "question": "The kids ran into the XXXXX.",
        "answer": "field",
    }
    rec[field] = value
    assert _normalize_cbt_record(rec, _src("cbt")) is None


def test_bookcorpus_lambada_basic():
    rec = {
        "text": (
            "She walked along the cobblestone street under the dim lamplight, "
            "her footsteps echoing softly. The night air was crisp and the "
            "shop windows glowed faintly behind their iron grilles. "
            "She paused at the corner and looked back, but the alley was empty."
        ),
    }
    result = _normalize_bookcorpus_lambada_record(rec, _src("bookcorpus_lambada"))
    assert result is not None
    messages, _meta = result
    user = messages[0]["content"]
    assistant = messages[1]["content"]

    assert assistant.strip() == "empty"
    assert user.endswith(" ")
    assert not user.endswith("  ")
    # The final word "empty" must NOT appear in the prompt — only the prefix.
    # (Whole-word check, not substring, since "empty" could appear elsewhere.)
    assert "empty " not in user
    assert not user.endswith("empty ")


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("text", "", "empty text"),
        ("text", "Too short to use.", "below minimum word count"),
        ("text", None, "missing text"),
    ],
    ids=["empty", "short", "none"],
)
def test_bookcorpus_lambada_rejects_invalid(field, value, reason):
    rec = {
        "text": (
            "She walked along the cobblestone street under the dim lamplight, "
            "her footsteps echoing softly. The night air was crisp and the "
            "shop windows glowed faintly behind their iron grilles. "
            "She paused at the corner."
        ),
    }
    rec[field] = value
    assert _normalize_bookcorpus_lambada_record(rec, _src("bookcorpus_lambada")) is None


def test_flan_mc_extracts_letter_from_response():
    """OpenOrca-style: question has MC structure, response has reasoning + final letter."""
    rec = {
        "system_prompt": "",
        "question": (
            "What is 2 + 2?\n"
            "A) 3\n"
            "B) 4\n"
            "C) 5\n"
            "D) 6\n"
            "Answer:"
        ),
        "response": "We add 2 and 2 to get 4. The answer is B.",
    }
    result = _normalize_flan_mc_record(rec, _src("flan_mc"))
    assert result is not None
    messages, _meta = result
    assert messages[1]["content"] == "B"
    user = messages[0]["content"]
    # The question is preserved verbatim — we don't re-render the MC structure.
    assert "A) 3" in user
    assert user.endswith("Answer:")


def test_flan_mc_rejects_non_mc_questions():
    rec = {"question": "What is the capital of France?", "response": "Paris."}
    assert _normalize_flan_mc_record(rec, _src("flan_mc")) is None


def test_flan_mc_rejects_unparseable_response():
    rec = {
        "question": "Q?\nA) x\nB) y\nAnswer:",
        "response": "I'm not sure but maybe both could be valid...",
    }
    assert _normalize_flan_mc_record(rec, _src("flan_mc")) is None


def test_flan_mc_extracts_letter_from_leading_letter_response():
    """Some FLAN-MC responses are just the bare letter."""
    rec = {
        "question": "Q?\nA) yes\nB) no\nC) maybe\nD) unsure\nAnswer:",
        "response": "C",
    }
    result = _normalize_flan_mc_record(rec, _src("flan_mc"))
    assert result is not None
    messages, _meta = result
    assert messages[1]["content"] == "C"
