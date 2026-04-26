"""Direct Preference Optimization (DPO) for ParrotLLM — VL08 implementation.

Reference material:
- VL08_RLHF_DPO.pdf (lecture, Handschuh) in docs/post_training/course_materials/
- Rafailov et al. 2023, *Direct Preference Optimization: Your Language Model
  is Secretly a Reward Model*, NeurIPS, arXiv:2305.18290 (slide 33's
  derivation is straight from this paper)
- EX08_DPO.ipynb — the toy reference implementation

The defining identity, VL08 slide 33:

    L_DPO = -log σ(β [log(π_θ(y_w|x)/π_ref(y_w|x))
                     - log(π_θ(y_l|x)/π_ref(y_l|x))])

i.e. the policy raises the log-prob ratio of the chosen response over
the rejected response, both measured relative to the frozen reference.
β is the implicit KL temperature — VL08 slide 21's "invisible leash".

Module layout (parallels src/post_training/sft/):

- `template.py`  — reuses the SFT Alpaca template (slide 32 critical rule)
- `collator.py`  — pads chosen+rejected, builds -100 masks for both halves
- `data.py`      — load HF preference datasets, normalise to {prompt, chosen, rejected}
- `trainer.py`   — load SFT as policy + frozen ref, DPO loss, AdamW step

Vanilla PyTorch implementation per the course constraint. HuggingFace is
used only for the GPT-2 tokenizer (already extended with `<|pad|>`) and
`datasets.load_dataset` for HF Hub data — no TRL.
"""

from src.post_training.dpo.template import (
    DPO_DEFAULT_TEMPLATE,
    format_dpo_prompt,
)
from src.post_training.dpo.collator import DPOCollator, IGNORE_INDEX
from src.post_training.dpo.data import build_dpo_datasets

__all__ = [
    "DPOCollator",
    "DPO_DEFAULT_TEMPLATE",
    "IGNORE_INDEX",
    "build_dpo_datasets",
    "format_dpo_prompt",
]
