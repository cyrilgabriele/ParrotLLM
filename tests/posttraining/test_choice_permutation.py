"""Tests for the deterministic choice-order permutation helper."""

from __future__ import annotations

from collections import Counter

from src.posttraining.prepare import _permute_choices


def test_permute_choices_is_deterministic_per_seed_key():
    choices = ["alpha", "beta", "gamma", "delta"]
    answer_index = 1
    seed_key = "example_42"

    out_choices_a, out_index_a = _permute_choices(choices, answer_index, seed_key)
    out_choices_b, out_index_b = _permute_choices(choices, answer_index, seed_key)

    assert out_choices_a == out_choices_b
    assert out_index_a == out_index_b


def test_permute_choices_preserves_gold_text():
    choices = ["alpha", "beta", "gamma", "delta"]
    answer_index = 2
    seed_key = "ex"

    out_choices, out_index = _permute_choices(choices, answer_index, seed_key)

    assert out_choices[out_index] == "gamma"
    assert sorted(out_choices) == sorted(choices)


def test_permute_choices_uniform_over_many_examples():
    """Across 4000 random keys, the gold index should be roughly uniform."""
    choices = ["a", "b", "c", "d"]
    counter: Counter[int] = Counter()
    for i in range(4000):
        _, idx = _permute_choices(choices, 0, f"key_{i}")
        counter[idx] += 1

    assert set(counter) == {0, 1, 2, 3}
    expected = 4000 / 4
    for count in counter.values():
        assert abs(count - expected) < 0.10 * expected, counter


def test_permute_choices_handles_binary():
    choices = ["yes", "no"]
    out_choices, out_index = _permute_choices(choices, 0, "binary_key_1")
    assert out_choices[out_index] == "yes"
    assert sorted(out_choices) == ["no", "yes"]


from collections import Counter as _Counter

from configs import SFTSourceConfig
from src.posttraining.prepare import _normalize_hellaswag_record


def _make_hellaswag_record(record_id: str, gold_index: int) -> dict:
    return {
        "ind": record_id,
        "ctx": f"A man is walking down the street. He",
        "endings": ["sees a dog.", "buys a coffee.", "trips on a curb.", "starts to fly."],
        "label": gold_index,
    }


def _make_source_cfg() -> SFTSourceConfig:
    return SFTSourceConfig(
        name="hs_test",
        loader="hellaswag",
        path="Rowan/hellaswag",
        split="train",
        target_examples=1,
    )


def test_hellaswag_normalizer_permutes_gold_letter():
    """When the source's gold is always index 0, after permutation the
    gold letter should be approximately uniform across A/B/C/D."""
    src = _make_source_cfg()
    letter_counts: _Counter[str] = _Counter()

    for i in range(800):
        rec = _make_hellaswag_record(record_id=str(i), gold_index=0)
        result = _normalize_hellaswag_record(rec, src)
        assert result is not None, f"normalizer rejected synthetic record {i}"
        messages, _meta = result
        gold_letter = messages[1]["content"]
        letter_counts[gold_letter] += 1

    assert set(letter_counts) == {"A", "B", "C", "D"}
    expected = 800 / 4
    for count in letter_counts.values():
        assert abs(count - expected) < 0.15 * expected, letter_counts
