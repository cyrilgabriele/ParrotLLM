from __future__ import annotations

import random

from tools.build_hybrid_mc_dpo_preferences import _to_preference


def test_dpo_mc_preference_builder_uses_correct_letter_as_chosen():
    row = {
        "instruction": "Context: Q\nA) red\nB) blue\nC) green\nD) yellow\nAnswer:",
        "response": " C",
        "source": "unit",
    }
    pref = _to_preference(row, rng=random.Random(0))
    assert pref is not None
    assert pref["prompt"] == row["instruction"]
    assert pref["chosen"] == " C"
    assert pref["rejected"].strip() in {"A", "B", "D"}
    assert pref["template"] == "raw"


def test_dpo_mc_preference_builder_rejects_missing_wrong_option():
    row = {
        "instruction": "Context: Q\nA) only\nAnswer:",
        "response": " A",
        "source": "unit",
    }
    assert _to_preference(row, rng=random.Random(0)) is None
