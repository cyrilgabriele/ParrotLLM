"""Padding + label-masking collator for DPO batches.

Each example yields TWO sequences — chosen and rejected — both rendered
as ``prompt + response + eos``. The collator pads them independently to
the longest sequence in their respective half and produces:

    chosen_input_ids   (B, T_c)   - token ids for the chosen completion
    chosen_labels      (B, T_c)   - -100 on prompt + pad, ids on response
    chosen_attention_mask (B, T_c)
    rejected_input_ids (B, T_r)   - same layout for rejected
    rejected_labels    (B, T_r)
    rejected_attention_mask (B, T_r)

T_c and T_r can differ. We do NOT pad them to the same length — that
wastes compute and there is no requirement that they match. The trainer
runs separate forward passes for chosen and rejected anyway.

The label-masking convention is identical to SFT (VL07 slide 15) — the
same -100 sentinel, the same prompt-only mask boundary. This reuse means
the DPO trainer's per-token log-prob aggregation can use the same gather
+ mask pattern as SFT's CE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


IGNORE_INDEX = -100


@dataclass
class DPOCollator:
    """Batch collator for DPO preference pairs.

    Args:
        pad_token_id: tokenizer's `<|pad|>` id (50257 for ParrotLLM).
        max_length: hard truncation ceiling per sequence. Same constraint
            as SFT (must be ≤ model.context_length = 1024).
        pad_to_multiple_of: optional throughput-tuning knob, multiples of
            8 light up Ampere+ tensor cores.
    """

    pad_token_id: int
    max_length: int = 1024
    pad_to_multiple_of: int | None = None

    def __call__(
        self, batch: Sequence[Mapping[str, list[int] | int]]
    ) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("DPOCollator received an empty batch.")

        chosen_pack = self._pack_half(
            [(ex["chosen_input_ids"], ex["chosen_prompt_length"]) for ex in batch],
        )
        rejected_pack = self._pack_half(
            [(ex["rejected_input_ids"], ex["rejected_prompt_length"]) for ex in batch],
        )
        return {
            "chosen_input_ids": chosen_pack["input_ids"],
            "chosen_labels": chosen_pack["labels"],
            "chosen_attention_mask": chosen_pack["attention_mask"],
            "rejected_input_ids": rejected_pack["input_ids"],
            "rejected_labels": rejected_pack["labels"],
            "rejected_attention_mask": rejected_pack["attention_mask"],
        }

    def _pack_half(
        self, items: list[tuple[list[int], int]]
    ) -> dict[str, torch.Tensor]:
        ids_list: list[list[int]] = []
        prompt_lens: list[int] = []
        for ids, p_len in items:
            ids_truncated = list(ids)[: self.max_length]
            ids_list.append(ids_truncated)
            prompt_lens.append(min(int(p_len), len(ids_truncated)))

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
            if p_len < L:
                labels[b, p_len:L] = torch.tensor(ids[p_len:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def count_supervised_tokens(labels: torch.Tensor) -> int:
    """Number of tokens contributing to the per-sequence log-prob sum.

    For DPO the per-sequence score is a SUM (not mean) of log-probs over
    response positions, so this count is informational only. Diagnostic
    use: confirm that prompt masking is wired correctly before committing
    to a long run.
    """
    return int((labels != IGNORE_INDEX).sum().item())
