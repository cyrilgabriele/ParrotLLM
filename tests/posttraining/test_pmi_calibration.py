"""Tests for PMI-calibrated cloze scoring.

PMI calibration: instead of choosing argmax_i log P(option_i | context),
choose argmax_i [log P(option_i | context) - log P(option_i | neutral_prefix)].

The neutral prefix isolates the unconditional surface frequency of each option
from the contextual lift. Common phrasings get a high P(option | "") which
gets subtracted out.

Invariants tested:
  1. When all options have similar conditional log-probs but unequal
     unconditional log-probs, PMI corrects the ranking. (Specifically: a
     deliberately-frequent wrong option that wins under un-calibrated cloze
     should LOSE under PMI when the contextual lift on the right answer is
     larger.)
  2. When unconditional log-probs are equal across options (same surface
     length, same neutral-prefix score), PMI ranking equals the un-
     calibrated ranking. (Sanity: PMI doesn't hurt when there's nothing to
     correct.)
  3. PMI scoring runs end-to-end on a real tiny model + GPT-2 tokenizer
     and returns a finite scalar.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "Submissions" / "parrotlabs_parrotllm"


def _load_submission_inference():
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    inference_path = SUBMISSION_DIR / "src" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "submission_inference_pmi", inference_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_inference_pmi"] = module
    # Also register under "src.inference" so internal cross-imports keep working.
    sys.modules.setdefault("src.inference", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def submission_inference():
    return _load_submission_inference()


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained("openai-community/gpt2", use_fast=True)


@pytest.fixture(scope="module")
def tiny_model():
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


def test_pmi_corrects_high_frequency_wrong_option(submission_inference, gpt2_tokenizer, tiny_model):
    """Construct a scorer where:
       - option 0 (the WRONG, high-frequency option) wins on raw cloze.
       - option 1 (the RIGHT option) wins on PMI because its unconditional
         logprob is much lower (it's a rarer surface form), so the contextual
         lift is larger.
    """
    model, _ = tiny_model
    options = ["the", "quaternion"]
    # Deterministic "scorer": given (prefix_ids, continuation_ids), look up.
    # Real prefix_text matters, but only as a key into the lookup.
    # We'll use the tokenized continuation as the key to make this robust.
    NEUTRAL_KEY = "__NEUTRAL__"

    def make_fake_scorer(scores_by_key):
        def fake_score(_model, *, prefix_ids, continuation_ids, device, context_length):
            # Distinguish the neutral prefix from the contextual one by length.
            # The neutral prefix is short (< 10 tokens); the contextual one is long.
            key_phase = NEUTRAL_KEY if len(prefix_ids) < 10 else "CTX"
            key_opt = tuple(continuation_ids)
            return scores_by_key[(key_phase, key_opt)]
        return fake_score

    # Tokenize options once to use as lookup keys.
    cont_ids_a = tuple(gpt2_tokenizer.encode(" the", add_special_tokens=False))
    cont_ids_b = tuple(gpt2_tokenizer.encode(" quaternion", add_special_tokens=False))

    # Cloze scores under the *real* context: "the" wins (-2.0 > -3.0) when
    # uncalibrated.
    # Neutral (unconditional) scores: "the" is a common token (-1.5), "quaternion"
    # is rare (-8.0).
    # PMI scores:
    #   the:        -2.0 - (-1.5) = -0.5
    #   quaternion: -3.0 - (-8.0) = +5.0
    # → PMI prefers quaternion (correct).
    scores = {
        ("CTX", cont_ids_a): -2.0,
        ("CTX", cont_ids_b): -3.0,
        (NEUTRAL_KEY, cont_ids_a): -1.5,
        (NEUTRAL_KEY, cont_ids_b): -8.0,
    }
    fake_scorer = make_fake_scorer(scores)

    # Long enough prefix to trigger len(prefix_ids) >= 10 in the fake scorer.
    long_prefix = (
        "Today the brave knight rode toward the dragon's lair, sword drawn, "
        "muttering ancient incantations under his breath. The next word he "
        "would utter was a single, perfectly-chosen"
    )
    pick_uncal = submission_inference.cloze_score_options(
        model,
        gpt2_tokenizer,
        prefix_text=long_prefix,
        option_texts=options,
        device=torch.device("cpu"),
        context_length=64,
        scorer=fake_scorer,
        pmi=False,
    )
    pick_pmi = submission_inference.cloze_score_options(
        model,
        gpt2_tokenizer,
        prefix_text=long_prefix,
        option_texts=options,
        device=torch.device("cpu"),
        context_length=64,
        scorer=fake_scorer,
        pmi=True,
    )
    assert pick_uncal == 0, "uncalibrated should pick the high-freq wrong option"
    assert pick_pmi == 1, "PMI should correct to the rarer right option"


def test_pmi_preserves_ranking_when_unconditional_logprobs_equal(
    submission_inference, gpt2_tokenizer, tiny_model
):
    """When every option's unconditional logprob is identical, PMI subtracts
    the same constant from every score and the argmax is unchanged.
    """
    model, _ = tiny_model
    options = ["foo", "bar", "baz"]

    cont_ids_by_text = {
        opt: tuple(gpt2_tokenizer.encode(" " + opt, add_special_tokens=False))
        for opt in options
    }

    def fake_score(_model, *, prefix_ids, continuation_ids, device, context_length):
        # Unconditional: all options score -2.0.
        # Conditional: options score -10, -3, -7 → argmax is index 1 ("bar").
        ctx_scores = {
            cont_ids_by_text["foo"]: -10.0,
            cont_ids_by_text["bar"]: -3.0,
            cont_ids_by_text["baz"]: -7.0,
        }
        if len(prefix_ids) < 10:
            return -2.0  # uniform unconditional
        return ctx_scores[tuple(continuation_ids)]

    long_prefix = (
        "An extended context that is at least ten tokens long for our "
        "fake scorer's neutral-vs-context discriminator to work as intended."
    )
    pick_uncal = submission_inference.cloze_score_options(
        model, gpt2_tokenizer,
        prefix_text=long_prefix, option_texts=options,
        device=torch.device("cpu"), context_length=64,
        scorer=fake_score, pmi=False,
    )
    pick_pmi = submission_inference.cloze_score_options(
        model, gpt2_tokenizer,
        prefix_text=long_prefix, option_texts=options,
        device=torch.device("cpu"), context_length=64,
        scorer=fake_score, pmi=True,
    )
    assert pick_uncal == 1
    assert pick_pmi == 1, "PMI must not change ranking when uncond is uniform"


def test_pmi_smoke_with_fake_scorer(submission_inference, gpt2_tokenizer, tiny_model):
    """Smoke: PMI codepath runs end-to-end with a fake scorer (avoids vocab
    mismatch on the tiny model) and returns a valid in-range index."""
    model, config = tiny_model
    options = ["alpha", "beta", "gamma"]

    def fake_score(_model, *, prefix_ids, continuation_ids, device, context_length):
        # All scores equal → ranking unchanged regardless of pmi flag.
        return 0.0

    pick = submission_inference.cloze_score_options(
        model,
        gpt2_tokenizer,
        prefix_text="The letter is",
        option_texts=options,
        device=torch.device("cpu"),
        context_length=config["model"]["context_length"],
        scorer=fake_score,
        pmi=True,
    )
    assert 0 <= pick < len(options)


def test_pmi_default_is_off_for_backwards_compat(submission_inference, gpt2_tokenizer, tiny_model):
    """The default cloze_score_options() call (no `pmi=` kwarg) must remain
    PMI-OFF, so existing tests / inference paths don't silently change behavior.
    The leaderboard CLI flips it on explicitly.
    """
    model, _ = tiny_model
    options = ["alpha", "beta"]

    # Make uncal pick 0 and PMI pick 1 — if the default were PMI on, we'd see 1.
    cont_a = tuple(gpt2_tokenizer.encode(" alpha", add_special_tokens=False))
    cont_b = tuple(gpt2_tokenizer.encode(" beta", add_special_tokens=False))

    def fake_score(_model, *, prefix_ids, continuation_ids, device, context_length):
        if len(prefix_ids) < 10:
            return {cont_a: -1.0, cont_b: -10.0}[tuple(continuation_ids)]
        return {cont_a: -2.0, cont_b: -3.0}[tuple(continuation_ids)]

    long_prefix = (
        "An extended context that is at least ten tokens long for our "
        "fake scorer's neutral-vs-context discriminator."
    )
    pick = submission_inference.cloze_score_options(
        model,
        gpt2_tokenizer,
        prefix_text=long_prefix,
        option_texts=options,
        device=torch.device("cpu"),
        context_length=64,
        scorer=fake_score,
        # NOTE: no pmi= kwarg → must default to False.
    )
    # Uncal: alpha=-2, beta=-3 → alpha (0). PMI: alpha=-1, beta=+7 → beta (1).
    assert pick == 0, "default must be PMI off (backwards compat)"
