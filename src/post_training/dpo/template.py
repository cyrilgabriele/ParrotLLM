"""DPO prompt template — re-exports the SFT Alpaca template.

VL07 slide 32's "critical rule" is a system-wide invariant: the prompt
format used at training time MUST match every downstream stage including
DPO and inference. ParrotLLM committed to Alpaca at SFT, so DPO inherits
it unchanged.

This module is intentionally thin — it exists so the DPO package has a
named public surface (`format_dpo_prompt`) that callers can import
without reaching into `src.post_training.sft.template`. Should DPO ever
need a divergent template (it should not), the indirection here is the
single place to change.
"""

from __future__ import annotations

from src.post_training.sft.template import (
    DEFAULT_ALPACA_TEMPLATE as DPO_DEFAULT_TEMPLATE,
    format_sft_prompt as format_dpo_prompt,
)

__all__ = ["DPO_DEFAULT_TEMPLATE", "format_dpo_prompt"]
