"""Tests for the leaderboard submission's inference helpers and main.generate()."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "Submissions" / "PikoGPPT_ParrotLabs"


def _load_submission_module(name: str, path: Path):
    """Load a module from the submission folder so tests can import its main/inference.

    Registering the module in sys.modules BEFORE exec is required for @dataclass
    annotations: dataclasses look up cls.__module__ in sys.modules to evaluate
    forward refs, and that lookup fails with AttributeError if the module isn't
    registered.

    The submission's main.py does `from src.inference import ...`. Python caches
    a `src` module from the FIRST `src/` package it finds on sys.path. If the
    project's `src/` was imported earlier in the test session, that cached
    module wins and `src.inference` won't exist (the project's src has no
    inference.py). To make this loader robust regardless of test order:
      1) Pre-load the submission's src.inference into sys.modules under both
         its real name and as `src.inference` so main.py's import resolves.
      2) When main.py is the target, also register `submission_main` so
         dataclass annotations work (above).
    """
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))

    # Pre-load src.inference so main.py's `from src.inference import ...` works
    # even when the project's `src` package is already cached in sys.modules.
    inference_path = SUBMISSION_DIR / "src" / "inference.py"
    if "src.inference" not in sys.modules and inference_path.exists():
        inf_spec = importlib.util.spec_from_file_location("src.inference", inference_path)
        inf_module = importlib.util.module_from_spec(inf_spec)
        sys.modules["src.inference"] = inf_module
        inf_spec.loader.exec_module(inf_module)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def test_score_continuation_logprob_is_finite(
    submission_inference, gpt2_tokenizer, tiny_model
):
    """Smoke: scoring runs end-to-end and returns a finite scalar.
    Exact values depend on model init; we just need monotonic, finite output."""
    model, config = tiny_model
    # Use small token ids that fit the tiny model's vocab=64.
    prefix_ids = [1, 2, 3, 4]
    cont_a = [5, 6]
    cont_b = [7, 8, 9]

    score_a = submission_inference.score_continuation_logprob(
        model,
        prefix_ids=prefix_ids,
        continuation_ids=cont_a,
        device=torch.device("cpu"),
        context_length=config["model"]["context_length"],
    )
    score_b = submission_inference.score_continuation_logprob(
        model,
        prefix_ids=prefix_ids,
        continuation_ids=cont_b,
        device=torch.device("cpu"),
        context_length=config["model"]["context_length"],
    )
    assert torch.isfinite(torch.tensor([score_a, score_b])).all()


def test_score_continuation_logprob_empty_continuation_returns_zero(
    submission_inference, tiny_model
):
    model, config = tiny_model
    score = submission_inference.score_continuation_logprob(
        model,
        prefix_ids=[1, 2, 3],
        continuation_ids=[],
        device=torch.device("cpu"),
        context_length=config["model"]["context_length"],
    )
    assert score == 0.0


def test_cloze_score_options_picks_best(submission_inference, gpt2_tokenizer, tiny_model):
    """Inject a fake scorer so we know which index 'wins' regardless of model state."""
    model, _ = tiny_model
    options = ["alpha", "beta", "gamma", "delta"]
    fake_scores = {0: -10.0, 1: -5.0, 2: -1.0, 3: -3.0}
    call_count = {"n": 0}

    def fake_score(_model, *, prefix_ids, continuation_ids, device, context_length):
        i = call_count["n"]
        call_count["n"] += 1
        return fake_scores[i]

    pick = submission_inference.cloze_score_options(
        model,
        gpt2_tokenizer,
        prefix_text="The best letter is:",
        option_texts=options,
        device=torch.device("cpu"),
        context_length=64,
        scorer=fake_score,
    )
    assert pick == 2
    assert call_count["n"] == 4


def test_dispatch_lambada_rstrips_prompt(submission_main):
    """LAMBADA path must rstrip the trailing space before tokenizing."""
    rendered = submission_main.render_prompt_for_inference(
        raw_prompt=LAMBADA_PROMPT,
        template="alpaca",
        system_prompt="ignored-in-leaderboard-mode",
        leaderboard=True,
    )
    assert rendered.kind == "lambada"
    assert not rendered.text.endswith(" ")
    assert rendered.text == LAMBADA_PROMPT.rstrip()


def test_dispatch_mc_uses_full_prompt(submission_main):
    rendered = submission_main.render_prompt_for_inference(
        raw_prompt=HELLASWAG_PROMPT,
        template="alpaca",
        system_prompt="ignored",
        leaderboard=True,
    )
    assert rendered.kind == "mc"
    assert rendered.mc_options == [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof.",
    ]


def test_dispatch_chat_applies_alpaca_wrap(submission_main):
    rendered = submission_main.render_prompt_for_inference(
        raw_prompt=CHAT_PROMPT,
        template="alpaca",
        system_prompt="You are ParrotLLM, a helpful assistant.",
        leaderboard=False,
    )
    assert rendered.kind == "chat"
    assert "### Instruction:" in rendered.text
    assert "### Response:" in rendered.text


def test_dispatch_raw_template_skips_wrap(submission_main):
    rendered = submission_main.render_prompt_for_inference(
        raw_prompt=CHAT_PROMPT,
        template="raw",
        system_prompt="ignored",
        leaderboard=False,
    )
    assert rendered.kind == "chat"
    assert rendered.text == CHAT_PROMPT


def test_constrained_fallback_emits_allowed_letter(
    submission_main, submission_inference, gpt2_tokenizer, tiny_model
):
    """When asked to generate with allowed_first_token_ids set, the first
    new token MUST be one of the allowed ids regardless of model state."""
    model, config = tiny_model
    allowed = submission_inference.letter_token_ids(gpt2_tokenizer, ["A", "B", "C", "D"])
    # Pick an arbitrary prompt; we only check that the constraint binds.
    input_ids = [1, 2, 3]
    idx = torch.tensor([input_ids], dtype=torch.long)
    out = submission_main.generate(
        model,
        idx,
        max_new_tokens=1,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        context_length=config["model"]["context_length"],
        allowed_first_token_ids=allowed,
    )
    # tiny_model has vocab=64; many letter ids in GPT-2 are outside that range
    # so we relax to: the picked id is the highest-scoring id within `allowed`
    # that is < vocab_size. We just confirm the picked id is in the allowed set
    # AND inside the model vocab (guarding against the mask producing nonsense).
    new_token = int(out[0, -1].item())
    assert new_token in allowed
    assert 0 <= new_token < config["model"]["vocab_size"]


def test_dispatch_non_leaderboard_keeps_alpaca_wrap_for_mc(submission_main):
    """Outside leaderboard mode, the user might be testing chat — even an
    MC-shaped prompt should be alpaca-wrapped so it follows the chat path."""
    rendered = submission_main.render_prompt_for_inference(
        raw_prompt=HELLASWAG_PROMPT,
        template="alpaca",
        system_prompt="You are ParrotLLM, a helpful assistant.",
        leaderboard=False,
    )
    assert rendered.kind == "chat"
    assert "### Instruction:" in rendered.text
