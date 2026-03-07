"""Adapter for the standard Hugging Face GPT-2 checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class HuggingFaceGPT2(nn.Module):
    """Loads `openai-community/gpt2` for quick inference tests."""

    def __init__(self, model_name: str = "openai-community/gpt2") -> None:
        super().__init__()
        self.model_name = model_name
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.context_length = getattr(self.hf_model.config, "n_positions", 1024)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.hf_model(input_ids=idx, labels=targets)
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        return logits, loss
