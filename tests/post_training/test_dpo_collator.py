"""Tests for the DPO padding + masking collator.

The collator pads chosen and rejected halves INDEPENDENTLY (their token
counts can differ — there's no requirement that the chosen and rejected
completions have the same length). For each half, it applies the same
-100 mask convention as SFT (VL07 slide 15) on prompt + pad positions.
"""

from __future__ import annotations

import torch

from src.post_training.dpo.collator import (
    DPOCollator, IGNORE_INDEX, count_supervised_tokens,
)


def _ex(chosen_ids: list[int], chosen_p: int,
        rejected_ids: list[int], rejected_p: int) -> dict:
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_prompt_length": chosen_p,
        "rejected_input_ids": rejected_ids,
        "rejected_prompt_length": rejected_p,
    }


def test_collator_returns_six_tensors_with_aligned_shapes():
    coll = DPOCollator(pad_token_id=0)
    batch = [
        _ex([1, 2, 3, 4, 5], 2, [6, 7, 8], 1),
        _ex([9, 10, 11], 1, [12, 13, 14, 15, 16, 17], 2),
    ]
    out = coll(batch)
    assert set(out.keys()) == {
        "chosen_input_ids", "chosen_labels", "chosen_attention_mask",
        "rejected_input_ids", "rejected_labels", "rejected_attention_mask",
    }
    # Shapes within a half match; chosen and rejected can differ.
    assert out["chosen_input_ids"].shape == out["chosen_labels"].shape
    assert out["chosen_input_ids"].shape == out["chosen_attention_mask"].shape
    assert out["rejected_input_ids"].shape == out["rejected_labels"].shape


def test_collator_pads_chosen_and_rejected_independently():
    coll = DPOCollator(pad_token_id=0)
    batch = [
        _ex([1, 2, 3], 1, [4, 5, 6, 7, 8, 9], 1),
        _ex([1, 2], 1, [3, 4], 1),
    ]
    out = coll(batch)
    assert out["chosen_input_ids"].shape == (2, 3)
    assert out["rejected_input_ids"].shape == (2, 6)


def test_collator_masks_chosen_prompt_tokens():
    coll = DPOCollator(pad_token_id=0)
    batch = [_ex([10, 11, 12, 13, 14], chosen_p=3,
                 rejected_ids=[20, 21], rejected_p=1)]
    out = coll(batch)
    # chosen prompt positions 0,1,2 → -100; response 3,4 → ids
    assert out["chosen_labels"][0, :3].tolist() == [IGNORE_INDEX] * 3
    assert out["chosen_labels"][0, 3:].tolist() == [13, 14]


def test_collator_masks_rejected_prompt_tokens():
    coll = DPOCollator(pad_token_id=0)
    batch = [_ex([1, 2], 1, [99, 50, 51], 1)]
    out = coll(batch)
    assert out["rejected_labels"][0].tolist() == [IGNORE_INDEX, 50, 51]


def test_collator_masks_pad_positions():
    coll = DPOCollator(pad_token_id=0)
    batch = [_ex([1, 2, 3, 4, 5], 1, [10, 11, 12], 1),
             _ex([6, 7], 1, [20, 21, 22, 23, 24], 1)]
    out = coll(batch)
    # second row chosen padded at 2,3,4
    assert out["chosen_input_ids"][1, 2:].tolist() == [0, 0, 0]
    assert out["chosen_labels"][1, 2:].tolist() == [IGNORE_INDEX] * 3
    assert out["chosen_attention_mask"][1, 2:].tolist() == [0, 0, 0]


def test_collator_truncates_to_max_length():
    coll = DPOCollator(pad_token_id=0, max_length=4)
    batch = [_ex([1, 2, 3, 4, 5, 6, 7], 2, [10, 11, 12, 13, 14, 15, 16, 17], 2)]
    out = coll(batch)
    assert out["chosen_input_ids"].shape[1] == 4
    assert out["rejected_input_ids"].shape[1] == 4


def test_count_supervised_matches_per_half():
    coll = DPOCollator(pad_token_id=0)
    batch = [_ex([1, 2, 3, 4, 5], 2, [10, 11], 1)]
    out = coll(batch)
    # chosen: 5-2 = 3 supervised; rejected: 2-1 = 1 supervised
    assert count_supervised_tokens(out["chosen_labels"]) == 3
    assert count_supervised_tokens(out["rejected_labels"]) == 1


def test_collator_pad_to_multiple_of():
    coll = DPOCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = [_ex([1, 2, 3], 1, [4, 5], 1)]
    out = coll(batch)
    # Both halves padded up to next multiple of 8.
    assert out["chosen_input_ids"].shape[1] == 8
    assert out["rejected_input_ids"].shape[1] == 8
