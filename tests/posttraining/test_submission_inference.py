"""Tests for the leaderboard submission's inference helpers and main.generate()."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "Submissions" / "PikoGPPT_ParrotLabs"


def _load_submission_module(name: str, path: Path):
    """Load a module from the submission folder so tests can import its main/inference."""
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def submission_main():
    return _load_submission_module("submission_main", SUBMISSION_DIR / "main.py")


@pytest.fixture(scope="module")
def tiny_model():
    """A 2-layer ParrotLLM with vocab=64. Deterministic via fixed seed."""
    from src.model.transformer import ParrotLLM  # type: ignore[import-not-found]

    torch.manual_seed(0)
    config = {
        "model": {
            "vocab_size": 64,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 64,
            "context_length": 64,
            "dropout": 0.0,
            "bias": False,
            "rope_theta": 10000.0,
        }
    }
    model = ParrotLLM(config)
    model.eval()
    return model, config


def test_generate_stops_on_eos(submission_main, tiny_model):
    model, config = tiny_model
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    # Pick the argmax of the next-token distribution as our "EOS" — generate()
    # MUST then emit exactly one new token and stop, regardless of max_new_tokens.
    with torch.no_grad():
        logits, _ = model(idx)
    eos_id = int(logits[0, -1].argmax().item())

    out = submission_main.generate(
        model,
        idx,
        max_new_tokens=10,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        context_length=config["model"]["context_length"],
        eos_token_id=eos_id,
    )
    # 3 prompt tokens + 1 EOS = 4 total
    assert out.shape == (1, 4), f"expected shape (1, 4), got {tuple(out.shape)}"
    assert int(out[0, -1].item()) == eos_id


def test_generate_respects_allowed_first_token_ids(submission_main, tiny_model):
    model, config = tiny_model
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    allowed = [10, 20, 30]
    out = submission_main.generate(
        model,
        idx,
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        context_length=config["model"]["context_length"],
        allowed_first_token_ids=allowed,
    )
    # First newly emitted token must be one of the allowed ids.
    assert int(out[0, 3].item()) in allowed
