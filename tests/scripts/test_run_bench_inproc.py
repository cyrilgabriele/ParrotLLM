"""Tests for the in-process bench harness ``tools/run_bench_inproc.py``.

These tests exercise the harness's internal helpers without loading a real
checkpoint, and run the full per-example loop on a tiny in-memory model and
3-example slices of each benchmark JSONL. The point is to lock in the harness
CLI/data plumbing so future refactors don't break parsing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = PROJECT_ROOT / "tools" / "run_bench_inproc.py"
SUBMISSION_DIR = PROJECT_ROOT / "Submissions" / "PikoGPPT_ParrotLabs"
BENCH_ROOT = PROJECT_ROOT / "external" / "PikoGPT_Leaderboard" / "leaderboard" / "benchmarks"


@pytest.fixture(scope="module")
def harness():
    # Mirror the loader pattern from test_submission_inference.py: pre-register
    # src.inference under the submission's path so harness imports resolve to
    # the submission, not the project's src/ package.
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    inference_path = SUBMISSION_DIR / "src" / "inference.py"
    if "src.inference" not in sys.modules:
        spec = importlib.util.spec_from_file_location("src.inference", inference_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["src.inference"] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("harness_inproc_test", HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["harness_inproc_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_mc_letter_matches_runner(harness):
    # Mirrors leaderboard/run_benchmarks.py:parse_mc_letter exactly.
    assert harness.parse_mc_letter("A", {"A", "B", "C", "D"}) == "A"
    assert harness.parse_mc_letter(" b\n", {"A", "B"}) == "B"
    assert harness.parse_mc_letter("Z", {"A", "B"}) is None
    assert harness.parse_mc_letter("", {"A"}) is None


def test_parse_lambada_word_matches_runner(harness):
    # Same normalization as the runner.
    assert harness.parse_lambada_word(" Hello.\nworld") == "hello"
    assert harness.parse_lambada_word("    ") == ""
    assert harness.parse_lambada_word(" \"signs\"") == "signs"


def test_resolve_benches_expands_all(harness):
    assert harness.resolve_benches(["all"]) == list(harness.PUBLIC_BENCHES)
    assert harness.resolve_benches(["hellaswag", "lambada"]) == ["hellaswag", "lambada"]
    # dedup preserves first-seen order
    assert harness.resolve_benches(["hellaswag", "hellaswag"]) == ["hellaswag"]


def test_default_bench_paths_exist(harness):
    for name, path in harness.DEFAULT_BENCH_PATHS.items():
        assert path.is_file(), f"missing bench data for {name}: {path}"


def test_run_example_smoke_with_tiny_model(harness):
    """Run the full per-example flow against a 2-layer ParrotLLM on 1 MC and
    1 LAMBADA example. We don't assert correctness — only that the harness
    glues together prompt rendering, scoring/decoding, and parsing without
    crashing on either path.
    """
    from src.model.transformer import ParrotLLM  # type: ignore[import-not-found]
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", use_fast=True)
    torch.manual_seed(0)
    config = {
        "model": {
            "vocab_size": tok.vocab_size,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 64,
            "context_length": 256,
            "dropout": 0.0,
            "bias": False,
            "rope_theta": 10000.0,
        }
    }
    model = ParrotLLM(config)
    model.eval()

    # Pull a small slice from each bench file.
    hella = json.loads(BENCH_ROOT.joinpath("hellaswag/cleaned/validation.jsonl").open().readline())
    lamb = json.loads(BENCH_ROOT.joinpath("lambada/cleaned/test.jsonl").open().readline())

    inference_mod, main_mod = harness._load_submission_modules()

    out_mc = harness.run_example(
        raw_prompt=hella["prompt"],
        bench="hellaswag",
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        context_length=config["model"]["context_length"],
        eos_id=tok.eos_token_id,
        inference_mod=inference_mod,
        main_mod=main_mod,
        pmi_enabled=False,
        mc_max_tokens=3,
        lambada_max_tokens=5,
    )
    assert isinstance(out_mc, str) and len(out_mc) >= 1
    assert out_mc[0].upper() in {"A", "B", "C", "D"}

    out_lamb = harness.run_example(
        raw_prompt=lamb["prompt"],
        bench="lambada",
        model=model,
        tokenizer=tok,
        device=torch.device("cpu"),
        context_length=config["model"]["context_length"],
        eos_id=tok.eos_token_id,
        inference_mod=inference_mod,
        main_mod=main_mod,
        pmi_enabled=False,
        mc_max_tokens=3,
        lambada_max_tokens=5,
    )
    assert isinstance(out_lamb, str)
