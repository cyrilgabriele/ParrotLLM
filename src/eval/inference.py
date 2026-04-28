"""Autoregressive generation and leaderboard inference."""

import logging
import re
import sys

import torch

log = logging.getLogger("parrotllm.inference")

from configs import ProjectConfig
from src.model import HuggingFaceGPT2, ParrotLLM
from src.utils import build_tokenizer, maybe_load_hf_token


def _build_ngram_map(seq: list[int], n: int) -> dict[tuple, set[int]]:
    """`(n-1)-prefix -> set of tokens that have followed it`. The standard
    no-repeat-n-gram tracking structure used by HuggingFace generate.
    """
    out: dict[tuple, set[int]] = {}
    if n <= 1:
        return out
    for i in range(n - 1, len(seq)):
        out.setdefault(tuple(seq[i - n + 1:i]), set()).add(seq[i])
    return out


def _forbid_ngram_repeat(logits: torch.Tensor, seq: list[int],
                         ngram_map: dict[tuple, set[int]], n: int) -> None:
    """Mask any token that would close a previously-seen n-gram."""
    if n <= 1 or len(seq) < n - 1:
        return
    prefix = tuple(seq[-(n - 1):])
    forbidden = ngram_map.get(prefix)
    if forbidden:
        logits[:, list(forbidden)] = float("-inf")


@torch.no_grad()
def generate_stream(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    context_length: int = 1024,
    eos_token_id: int | None = None,
    no_repeat_ngram_size: int = 0,
):
    """Token-streaming counterpart to ``generate``.

    Yields each newly-generated token id (as a 0-D Python int) one at a
    time so the UI can display tokens as soon as they are produced.
    Stops on EOS like ``generate``; honours the same sampling controls.

    ``no_repeat_ngram_size`` (default 0 = off): if > 1, forbids any token
    that would close an n-gram already present in the prompt+generated
    sequence. Standard fix for the small-model "...blue with a blue with
    a blue..." failure mode; matches HuggingFace generate semantics.
    """
    model.eval()
    seq = idx[0].tolist()
    ngram_map = _build_ngram_map(seq, no_repeat_ngram_size)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        _forbid_ngram_repeat(logits, seq, ngram_map, no_repeat_ngram_size)

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tok = int(next_token.item())
        seq.append(tok)
        if no_repeat_ngram_size > 1 and len(seq) >= no_repeat_ngram_size:
            ngram_map.setdefault(tuple(seq[-no_repeat_ngram_size:-1]), set()).add(tok)
        idx = torch.cat([idx, next_token], dim=1)
        yield tok
        if eos_token_id is not None and tok == eos_token_id:
            return


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    context_length: int = 1024,
    eos_token_id: int | None = None,
    allowed_first_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Autoregressive generation. temp=0 for greedy, temp>0 for sampling.

    If ``eos_token_id`` is provided, generation halts as soon as every
    sequence in the batch has emitted that token. SFT/DPO checkpoints are
    trained to emit EOS at the semantic end of a response; without this
    early-stop the loop runs to ``max_new_tokens`` and the post-EOS
    continuation is unconditioned garbage. Default ``None`` preserves the
    legacy "always run the full budget" behavior used by leaderboard /
    pretrain generation.

    If ``allowed_first_token_ids`` is provided, the very first generated
    token is forced to lie in that set (logits outside the set are masked
    to -inf at step 0 only). Used by leaderboard MC mode to guarantee the
    runner sees a letter as the first character of stdout.
    """
    model.eval()

    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, vocab)

        if step == 0 and allowed_first_token_ids is not None and allowed_first_token_ids:
            mask = torch.full_like(logits, float("-inf"))
            allowed = torch.tensor(sorted(allowed_first_token_ids), device=logits.device)
            mask[:, allowed] = logits[:, allowed]
            logits = mask

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return idx


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ParrotLLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


_MC_ANSWER_RE = re.compile(r"\nAnswer:\s*$")
_MC_LETTER_LINE_RE = re.compile(r"\n([A-Z])\) ")


def detect_mc_letters(prompt: str) -> list[str] | None:
    """Return the sorted list of allowed answer letters if `prompt` looks
    like a multiple-choice question, else None.

    Strict criterion: the prompt must end with ``"\\nAnswer:"`` (allowing
    trailing whitespace) AND contain at least two ``"\\n[A-Z]) "`` bullet
    lines. Both conditions are required so chat messages that *happen* to
    contain "Answer:" or a single "A) " bullet are not misclassified.
    """
    if not _MC_ANSWER_RE.search(prompt):
        return None
    letters = sorted(set(_MC_LETTER_LINE_RE.findall(prompt)))
    if len(letters) < 2:
        return None
    return letters


def mc_first_token_ids(tokenizer, letters: list[str]) -> set[int]:
    """Return the set of token ids that begin with one of `letters`.

    The leaderboard runner reads the first character of stdout (after
    ``lstrip``), so we want to allow any token whose decoded string's
    first non-space character is one of the allowed letters. Common
    GPT-2 BPE encodings: ``" A"`` (id 317), ``"A"`` (id 32), ``"\\nA"``
    (rare). We enumerate the obvious variants and take their first ids.
    """
    ids: set[int] = set()
    for letter in letters:
        for variant in (f" {letter}", letter):
            try:
                toks = tokenizer.encode(variant)
            except Exception:
                continue
            if toks:
                ids.add(int(toks[0]))
    return ids


def run_inference(
    project_config: ProjectConfig,
    *,
    checkpoint: str | None,
    device: torch.device,
    prompt: str | None,
    max_tokens_override: int | None,
    temperature_override: float | None,
    leaderboard: bool,
    mock_testing: bool,
    hf_token: str | None = None,
) -> None:
    inference_cfg = project_config.inference
    if inference_cfg is None:
        raise ValueError("Inference configuration missing; cannot run inference stage.")

    use_mock = mock_testing

    if use_mock:
        if hf_token:
            log.info("Using Hugging Face token from .env")
        model = HuggingFaceGPT2().to(device)
        mc = {"context_length": getattr(model, "context_length", 1024)}
        log.info("mock_testing enabled: using openai-community/gpt2")
    else:
        assert checkpoint, "--checkpoint required for inference"
        model, ckpt_config = load_model_from_checkpoint(checkpoint, device)
        mc = ckpt_config["model"]
    model.eval()

    tokenizer = build_tokenizer()

    input_text = prompt if prompt else "Parrot are amazing because"
    input_ids = tokenizer.encode(input_text)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    max_tokens = max_tokens_override or inference_cfg.max_tokens
    temperature = (
        temperature_override if temperature_override is not None else inference_cfg.temperature
    )
    top_k = inference_cfg.top_k
    top_p = inference_cfg.top_p

    allowed_first_ids: set[int] | None = None
    if leaderboard:
        letters = detect_mc_letters(input_text)
        if letters is not None:
            allowed_first_ids = mc_first_token_ids(tokenizer, letters)

    output = generate(
        model, idx, max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        context_length=mc["context_length"],
        allowed_first_token_ids=allowed_first_ids,
    )
    text = tokenizer.decode(output[0].tolist())

    if leaderboard:
        # leaderboard mode: ONLY generated text, no logging
        generated = tokenizer.decode(output[0, len(input_ids):].tolist())
        sys.stdout.write(generated)
    else:
        log.info(f"prompt: {input_text}")
        log.info(f"output: {text}")
