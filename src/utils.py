import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv


PAD_TOKEN = "<|pad|>"
DEFAULT_TOKENIZER_NAME = "openai-community/gpt2"


def maybe_load_hf_token(env_path: str | Path = Path(".env")) -> str | None:
    """Return the HF token if explicitly provided, otherwise None."""

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    path = Path(env_path)
    if not path.exists():
        return None
    
    return None


def build_tokenizer(
    *,
    add_prefix_space: bool = False,
    padding_side: str = "right",
    tokenizer_name: str | None = None,
) -> GPT2TokenizerFast:
    """Return a GPT-2 fast tokenizer with a consistent PAD token.
    
    Args:
        add_prefix_space: Whether to add space prefix for BPE.
        padding_side: Which side to pad ('left' or 'right').
        tokenizer_name: Tokenizer model name (default: uses DEFAULT_TOKENIZER_NAME).
    """

    def _load(name: str) -> GPT2TokenizerFast:
        return GPT2TokenizerFast.from_pretrained(
            name,
            use_fast=True,
            add_prefix_space=add_prefix_space,
        )

    model_name = tokenizer_name if tokenizer_name is not None else DEFAULT_TOKENIZER_NAME
    tokenizer = _load(model_name)

    tokenizer.padding_side = padding_side
    if tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN
    return tokenizer


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str != "auto":
        device = torch.device(device_str)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
