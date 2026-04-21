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
    render_example,
)
from src.post_training.sft.collator import SFTCollator
from src.post_training.sft.data import build_sft_datasets

__all__ = [
    "AlpacaTemplate",
    "DEFAULT_ALPACA_TEMPLATE",
    "render_example",
    "SFTCollator",
    "build_sft_datasets",
]
