"""Supervised Fine-Tuning (SFT) for ParrotLLM — VL07 implementation.

Reference material, all checked into `docs/post_training/`:
- VL07_Post_Training_SFT_GH_Edit.pdf (lecture, Wiegand)
- SFT.md (senior-PhD-level technical plan)
- sft_from_checkpoint_summary.md (Cyril's earlier SFT guide)

The core identity of this module is summarised on VL07 slide 14:

    SFT = fine-tuning on (instruction, response) pairs with the same
          CrossEntropy loss as pre-training, but only on response tokens.

Every submodule corresponds to one mechanical piece of that definition:

- `template.py`:  how (instruction, response) is rendered into a single string
- `collator.py`:  how the "loss only on response tokens" rule is enforced
                  at the batch level (labels = -100 on instruction positions)
- `data.py`:      where (instruction, response) pairs come from and how they
                  are decontaminated against the leaderboard test splits
- `trainer.py`:   the loop itself, initialised from a pretraining checkpoint
"""

from src.post_training.sft.template import (
    AlpacaTemplate,
    DEFAULT_ALPACA_TEMPLATE,
    format_sft_prompt,
    render_example,
)
from src.post_training.sft.collator import SFTCollator
from src.post_training.sft.data import build_sft_datasets

__all__ = [
    "AlpacaTemplate",
    "DEFAULT_ALPACA_TEMPLATE",
    "format_sft_prompt",
    "render_example",
    "SFTCollator",
    "build_sft_datasets",
]


# Inference contract (VL07 slide 32, "critical rule"):
#   from src.post_training.sft import format_sft_prompt
#   prompt = format_sft_prompt("What is the capital of France?")
#   ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#   out = generate(model, ids, max_new_tokens=128, ...)
#   completion = tokenizer.decode(out[0, ids.size(1):])
# Any code path that bypasses `format_sft_prompt` for inference against an
# SFT checkpoint will silently degrade output quality. See
# `src/post_training/sft/template.py::format_sft_prompt` for full
# rationale.
