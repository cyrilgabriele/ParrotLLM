"""Padding + label-masking collator for SFT batches.

This is the operational heart of VL07 slide 15 ("The Masked Loss: Only Learn
the Response") and slide 16 ("Token-Level Loss Masking: Detail View").

What the collator does:

1. Takes a list of pre-tokenised examples with fields
       ``input_ids``      — the full (prompt + response + eos) token ids
       ``prompt_length``  — how many tokens belong to the instruction prefix
2. Right-pads all sequences in the batch to a common length using
   ``pad_token_id``.
3. Produces a ``labels`` tensor such that
       labels[b, t] = -100                       if t < prompt_length[b]
                    = -100                       if t is a pad position
                    = input_ids[b, t]            otherwise
   i.e. loss is computed only on response tokens (VL07 slide 15).

Why label the response tokens with *their own* ids (and not a shifted copy):
our ``ParrotLLM.forward()`` does the next-token shift internally when it
sees HuggingFace-style ``labels`` (mirroring ``transformers``'
``PreTrainedModel.forward``). See ``src/model/transformer.py``.

Why ``-100`` and not a separate mask tensor: ``F.cross_entropy`` treats
``ignore_index=-100`` natively — no extra reduction logic, no branches in
the hot path. It also means a single ``labels`` tensor round-trips through
DDP, torch.compile, and mixed-precision autocast with no custom handling.
This convention comes from HuggingFace transformers and is the de-facto
standard referenced in every open SFT codebase (Alpaca-LoRA, TRL, OLMo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


IGNORE_INDEX = -100


@dataclass
class SFTCollator:
    """Batch collator: right-pad, build labels with -100 on prompt + pad tokens.

    Args:
        pad_token_id: tokenizer's `<|pad|>` id (ParrotLLM adds exactly one;
            by default `50257` since vocab is 50258 with `<|pad|>` appended).
        max_length: hard truncation ceiling. Must not exceed the model's
            context length (1024 per the PikoGPT fact-sheet constraint).
            Examples longer than this are truncated from the right — the
            trailing part of the response is dropped rather than the
            instruction, on the assumption that shorter truncated responses
            are more useful than missing-instruction examples.
        pad_to_multiple_of: optional. Padding to a multiple of 8 (or 16) is
            a small throughput win on Ampere+ tensor cores. Safe default
            None; callers can set 8 when training on CUDA.
    """

    pad_token_id: int
    max_length: int = 1024
    pad_to_multiple_of: int | None = None

    def __call__(self, batch: Sequence[Mapping[str, list[int] | int]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("SFTCollator received an empty batch.")

        ids_list: list[list[int]] = []
        prompt_lens: list[int] = []
        for ex in batch:
            ids = list(ex["input_ids"])[: self.max_length]
            p_len = min(int(ex["prompt_length"]), len(ids))
            ids_list.append(ids)
            prompt_lens.append(p_len)

        max_len = max(len(x) for x in ids_list)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        B = len(ids_list)
        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        for b, (ids, p_len) in enumerate(zip(ids_list, prompt_lens)):
            L = len(ids)
            input_ids[b, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[b, :L] = 1
            # Mask instruction tokens (positions < p_len) and pad positions
            # (positions >= L). Response tokens (p_len <= t < L) keep their
            # own ids as labels — VL07 slide 15.
            if p_len < L:
                labels[b, p_len:L] = torch.tensor(ids[p_len:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def count_supervised_tokens(labels: torch.Tensor) -> int:
    """Return the number of tokens that contribute to the loss.

    Useful in smoke tests: for a correctly-masked Alpaca batch we expect
    this count to equal ``sum(len(response) in tokens)`` across the batch.
    If it drops to 0 or spikes near sequence length * batch_size, the mask
    is wrong — one of VL07 slide 16's failure modes.
    """
    return int((labels != IGNORE_INDEX).sum().item())
