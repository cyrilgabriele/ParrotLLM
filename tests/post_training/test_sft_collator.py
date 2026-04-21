"""Tests for the SFT padding + masking collator.

VL07 slide 15 defines SFT loss as: CrossEntropy on RESPONSE tokens only.
Slide 16 shows the failure mode when masking is absent (30-70% of gradient
mass leaks into instruction tokens). These tests operationalise the
invariants that masking must preserve:

1. Instruction tokens are masked (labels = -100).
2. Response tokens are kept (labels = their own ids).
3. Padding is masked (labels = -100, attention_mask = 0).
4. The collator shape matches (input_ids.shape == labels.shape ==
   attention_mask.shape).
"""

from __future__ import annotations

import torch

from src.post_training.sft.collator import IGNORE_INDEX, SFTCollator, count_supervised_tokens


def _make_example(ids: list[int], prompt_length: int) -> dict:
    return {"input_ids": ids, "prompt_length": prompt_length}


def test_collator_shapes_and_pad_token():
    coll = SFTCollator(pad_token_id=0)
    batch = [_make_example([1, 2, 3, 4, 5], 2),
             _make_example([6, 7, 8], 1)]
    out = coll(batch)
    assert out["input_ids"].shape == out["labels"].shape == out["attention_mask"].shape
    assert out["input_ids"].shape == (2, 5)
    # First row should not be padded; second row tail positions should be pad.
    assert out["input_ids"][1, 3:].tolist() == [0, 0]
    assert out["attention_mask"][1, 3:].tolist() == [0, 0]


def test_collator_masks_instruction_tokens():
    """Positions before prompt_length must be -100 in labels."""
    coll = SFTCollator(pad_token_id=0)
    batch = [_make_example([10, 11, 12, 13, 14], prompt_length=3)]
    out = coll(batch)
    # Instruction positions 0, 1, 2 → labels = -100
    assert out["labels"][0, :3].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    # Response positions 3, 4 → labels = their own ids (VL07 slide 15).
    assert out["labels"][0, 3:].tolist() == [13, 14]


def test_collator_masks_padding_tokens():
    coll = SFTCollator(pad_token_id=0)
    batch = [_make_example([1, 2, 3, 4, 5], 2),
             _make_example([6, 7, 8], 1)]
    out = coll(batch)
    # Second row: positions 3, 4 are pad, must have label = -100.
    assert out["labels"][1, 3:].tolist() == [IGNORE_INDEX, IGNORE_INDEX]


def test_count_supervised_tokens_matches_expected():
    coll = SFTCollator(pad_token_id=0)
    # Example 1: 5 tokens, 2 prompt, 3 response.
    # Example 2: 3 tokens, 1 prompt, 2 response.
    # Total supervised = 3 + 2 = 5.
    batch = [_make_example([1, 2, 3, 4, 5], 2),
             _make_example([6, 7, 8], 1)]
    out = coll(batch)
    assert count_supervised_tokens(out["labels"]) == 5


def test_collator_respects_max_length_truncation():
    coll = SFTCollator(pad_token_id=0, max_length=4)
    batch = [_make_example([1, 2, 3, 4, 5, 6, 7], prompt_length=2)]
    out = coll(batch)
    assert out["input_ids"].shape[1] == 4
    # Prompt mask still at first 2 positions; positions 2,3 are response.
    assert out["labels"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, 4]


def test_collator_handles_zero_response_example_gracefully():
    """Corner case: prompt_length >= len(ids). No response → 0 supervised."""
    coll = SFTCollator(pad_token_id=0)
    batch = [_make_example([1, 2, 3], prompt_length=3)]
    out = coll(batch)
    # No positions should be supervised.
    assert count_supervised_tokens(out["labels"]) == 0


def test_collator_pad_to_multiple_of():
    coll = SFTCollator(pad_token_id=0, pad_to_multiple_of=8)
    batch = [_make_example([1, 2, 3], prompt_length=1)]
    out = coll(batch)
    # Length should be rounded up to 8.
    assert out["input_ids"].shape[1] == 8
    # Positions 3..7 are padding → labels = -100.
    assert out["labels"][0, 3:].tolist() == [IGNORE_INDEX] * 5
