"""Tests for the v9 hybrid MC mix builder helpers.

These tests avoid Hugging Face network access and focus on the important
contract: raw MC examples can be rendered with explicit answer-letter control.
"""

from __future__ import annotations

import random
from collections import Counter

from tools.build_hybrid_mc_sft_mix import MCItem, _render_raw_mc, _sample_rendered


def test_render_raw_mc_can_force_answer_letter():
    item = MCItem(
        source="unit",
        stem="Which option is correct?",
        choices=("correct", "wrong one", "wrong two", "wrong three"),
        answer_idx=0,
    )

    row = _render_raw_mc(item, rng=random.Random(1), force_answer_letter="C")

    assert row is not None
    assert "\nC) correct\n" in row["instruction"]
    assert row["response"] == " C"
    assert row["instruction"].endswith("\nAnswer:")


def test_sample_rendered_balances_four_way_answer_letters():
    items = [
        MCItem(
            source="unit",
            stem=f"Question {i}",
            choices=("correct", "wrong one", "wrong two", "wrong three"),
            answer_idx=0,
        )
        for i in range(8)
    ]

    rows = _sample_rendered(
        items,
        quota=8,
        rng=random.Random(2),
        decontam=set(),
        balance_four_way=True,
    )

    letters = Counter(row["response"].strip() for row in rows)
    assert letters == {"A": 2, "B": 2, "C": 2, "D": 2}
