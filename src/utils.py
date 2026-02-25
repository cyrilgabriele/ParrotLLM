import random

import numpy as np
import torch
import yaml
from transformers import GPT2TokenizerFast

DEFAULT_TOKENIZER_NAME = "openai-community/gpt2"
PAD_TOKEN = "<|pad|>"


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_tokenizer(
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    *,
    add_prefix_space: bool = False,
    padding_side: str = "right",
) -> GPT2TokenizerFast:
    """Return a GPT-2 fast tokenizer with a consistent PAD token."""

    def _load(name: str) -> GPT2TokenizerFast:
        return GPT2TokenizerFast.from_pretrained(
            name,
            use_fast=True,
            add_prefix_space=add_prefix_space,
        )

    name = tokenizer_name or DEFAULT_TOKENIZER_NAME
    try:
        tokenizer = _load(name)
    except OSError:
        if name != DEFAULT_TOKENIZER_NAME:
            raise
        tokenizer = _load("gpt2")

    tokenizer.padding_side = padding_side
    if tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN
    return tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
