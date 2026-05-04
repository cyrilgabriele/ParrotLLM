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


@pytest.fixture(scope="module")
def submission_inference():
    return _load_submission_module(
        "submission_inference", SUBMISSION_DIR / "src" / "inference.py"
    )


HELLASWAG_PROMPT = (
    "Context: A man is sitting on a roof. he\n"
    "A) is using wrap to wrap a pair of skis.\n"
    "B) is ripping level tiles off.\n"
    "C) is holding a rubik's cube.\n"
    "D) starts pulling up roofing on a roof.\n"
    "Answer:"
)
WINOGRANDE_PROMPT = (
    "Context: Sarah was a much better surgeon than Maria so _ always got the easier cases.\n"
    "A) Sarah\n"
    "B) Maria\n"
    "Answer:"
)
OPENBOOKQA_PROMPT = (
    "Question: Frilled sharks live deep in the ocean, which is why they are known as\n"
    "A) Deep sea animals\n"
    "B) fish\n"
    "C) Long Sea Fish\n"
    "D) Far Sea Animals\n"
    "Answer:"
)
LAMBADA_PROMPT = (
    "She walked through the door and her mouth curved in a confident grin, "
    "I don't care about "  # ← trailing space, no Answer:
)
CHAT_PROMPT = "Tell me a short joke about cats."


def test_detect_mc_prompt_hellaswag(submission_inference):
    parsed = submission_inference.detect_mc_prompt(HELLASWAG_PROMPT)
    assert parsed is not None
    stem, options, header = parsed
    assert header == "Context"
    assert stem.strip() == "A man is sitting on a roof. he"
    assert options == [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof.",
    ]


def test_detect_mc_prompt_winogrande(submission_inference):
    parsed = submission_inference.detect_mc_prompt(WINOGRANDE_PROMPT)
    assert parsed is not None
    stem, options, header = parsed
    assert header == "Context"
    assert "_" in stem
    assert options == ["Sarah", "Maria"]


def test_detect_mc_prompt_openbookqa(submission_inference):
    parsed = submission_inference.detect_mc_prompt(OPENBOOKQA_PROMPT)
    assert parsed is not None
    stem, options, header = parsed
    assert header == "Question"
    assert len(options) == 4


def test_detect_mc_prompt_rejects_lambada(submission_inference):
    assert submission_inference.detect_mc_prompt(LAMBADA_PROMPT) is None


def test_detect_mc_prompt_rejects_chat_with_answer_substring(submission_inference):
    # A chat prompt that *mentions* "Answer:" in passing must not be classified as MC.
    chat = "Hi! Could you give me the Answer: I am stuck on this problem."
    assert submission_inference.detect_mc_prompt(chat) is None


def test_is_lambada_shape(submission_inference):
    assert submission_inference.is_lambada_shape(LAMBADA_PROMPT) is True
    assert submission_inference.is_lambada_shape(HELLASWAG_PROMPT) is False
    assert submission_inference.is_lambada_shape(CHAT_PROMPT) is False


def test_wino_substitute(submission_inference):
    stem = "Sarah was a much better surgeon than Maria so _ always got the easier cases."
    head, tail = submission_inference.wino_substitute(stem, "Sarah")
    assert head == "Sarah was a much better surgeon than Maria so Sarah"
    assert tail == " always got the easier cases."


def test_wino_substitute_handles_missing_blank(submission_inference):
    # Falls back to ("<stem> <option>", "") when no _ is present.
    head, tail = submission_inference.wino_substitute("no blank here", "Sarah")
    assert head == "no blank here Sarah"
    assert tail == ""


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained("openai-community/gpt2", use_fast=True)


def test_letter_token_ids_includes_naked_letter(submission_inference, gpt2_tokenizer):
    ids = submission_inference.letter_token_ids(gpt2_tokenizer, ["A", "B", "C", "D"])
    plain = gpt2_tokenizer.encode("A", add_special_tokens=False)
    assert plain[0] in ids
    spaced = gpt2_tokenizer.encode(" A", add_special_tokens=False)
    assert spaced[0] in ids
    assert len(ids) == len(set(ids))
    allowed = {"A", "B", "C", "D"}
    for tid in ids:
        decoded = gpt2_tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        stripped = decoded.lstrip()
        assert stripped and stripped[0].upper() in allowed


def test_letter_token_ids_excludes_unrelated(submission_inference, gpt2_tokenizer):
    ids = set(submission_inference.letter_token_ids(gpt2_tokenizer, ["A", "B"]))
    c_ids = set(gpt2_tokenizer.encode(" C", add_special_tokens=False)[:1])
    one_ids = set(gpt2_tokenizer.encode(" 1", add_special_tokens=False)[:1])
    assert not (ids & c_ids)
    assert not (ids & one_ids)
