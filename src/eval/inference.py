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


def _apply_repetition_penalty(
    logits: torch.Tensor, seq_ids: list[int], penalty: float,
) -> torch.Tensor:
    """VL09 slide 25: ``z'_i = z_i / θ`` for tokens already in the sequence.

    Positive logits are divided by θ (push-down), negative logits are
    multiplied by θ (also push-down toward -∞). With θ=1.0 this is a no-op,
    which is the off-state. PikoGPT default per VL09: θ=1.1.

    Mutates ``logits`` in place and returns it for chaining.
    """
    if penalty == 1.0 or not seq_ids:
        return logits
    seen = torch.tensor(sorted(set(seq_ids)), dtype=torch.long, device=logits.device)
    slice_ = logits[:, seen]
    logits[:, seen] = torch.where(slice_ > 0, slice_ / penalty, slice_ * penalty)
    return logits


@torch.no_grad()
def generate_stream(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    context_length: int = 1024,
    eos_token_id: int | None = None,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.1,
    use_cache: bool = True,
):
    """Token-streaming counterpart to ``generate``.

    Yields each newly-generated token id (as a 0-D Python int) one at a
    time so the UI can display tokens as soon as they are produced.
    Stops on EOS like ``generate``; honours the same sampling controls.

    Defaults follow VL09 slide 30 ("PikoGPT default"): τ=0.8, top-p=0.9,
    rep.penalty=1.1.

    ``no_repeat_ngram_size`` (default 0 = off): if > 1, forbids any token
    that would close an n-gram already present in the sequence. Stronger
    than ``repetition_penalty`` (hard ban vs. logit divisor); the two
    compose. Matches HuggingFace generate semantics.

    ``repetition_penalty`` (VL09 slide 25): logits of tokens already in
    the sequence are divided by θ (positive) or multiplied by θ (negative),
    both pushing them down. θ=1.0 disables.

    ``use_cache``: if True (default) and the model exposes
    ``forward_with_cache``, generation runs with KV-cache for O(t) decode
    instead of O(t²) (VL09 slides 33–35). Falls back automatically for
    models without the cache method (e.g. HuggingFaceGPT2 wrapper).
    """
    model.eval()
    seq = idx[0].tolist()
    ngram_map = _build_ngram_map(seq, no_repeat_ngram_size)

    has_cache_path = use_cache and hasattr(model, "forward_with_cache")
    cache = None

    for step in range(max_new_tokens):
        if has_cache_path:
            # Prefill once on the full prompt, then feed only the last token
            # each subsequent step. Truncate the prefill to context_length.
            if cache is None:
                prefill_idx = idx[:, -context_length:]
                logits_full, cache = model.forward_with_cache(prefill_idx, cache=None)
                logits = logits_full[:, -1, :]
            else:
                last_tok = idx[:, -1:]
                logits_full, cache = model.forward_with_cache(last_tok, cache=cache)
                logits = logits_full[:, -1, :]
        else:
            idx_cond = idx[:, -context_length:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

        _apply_repetition_penalty(logits, seq, repetition_penalty)
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
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
) -> torch.Tensor:
    """Autoregressive generation. temp=0 for greedy, temp>0 for sampling.

    Defaults preserve the legacy leaderboard-mode behaviour (greedy, no
    repetition penalty); chat-style callers should pass
    ``temperature=0.8, repetition_penalty=1.1`` per VL09 slide 30.

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

    ``use_cache`` (VL09 slide 34): when True and the model exposes
    ``forward_with_cache``, decode runs in O(t) per step. Disable for
    parity tests or models without the cache path (e.g. HF GPT-2 wrapper).
    """
    model.eval()

    has_cache_path = (
        use_cache
        and hasattr(model, "forward_with_cache")
        and idx.size(0) == 1  # batched cached decode not supported here
    )
    cache = None
    seq_ids: list[int] = idx[0].tolist() if repetition_penalty != 1.0 else []

    for step in range(max_new_tokens):
        if has_cache_path:
            if cache is None:
                prefill_idx = idx[:, -context_length:]
                logits_full, cache = model.forward_with_cache(prefill_idx, cache=None)
                logits = logits_full[:, -1, :]
            else:
                logits_full, cache = model.forward_with_cache(idx[:, -1:], cache=cache)
                logits = logits_full[:, -1, :]
        else:
            idx_cond = idx[:, -context_length:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab)

        if repetition_penalty != 1.0:
            _apply_repetition_penalty(logits, seq_ids, repetition_penalty)

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
        if repetition_penalty != 1.0:
            seq_ids.append(int(next_token.item()))

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
_MC_OPTION_RE = re.compile(
    r"\n([A-Z])\)\s+(.+?)(?=\n[A-Z]\)\s+|\nAnswer:|\Z)",
    re.DOTALL,
)


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


def parse_mc_options(prompt: str) -> dict[str, str] | None:
    """Return ``{letter: option_text}`` for an MC prompt, or None if the
    prompt does not parse as MC. Used by the cloze-scoring path in
    ``--leaderboard`` mode.
    """
    matches = _MC_OPTION_RE.findall(prompt)
    if len(matches) < 2:
        return None
    return {letter: text.strip() for letter, text in matches}


def extract_mc_stem(prompt: str) -> str:
    """Return the bare question stem from an MC prompt (drops "Context:"/"Question:" prefix and the option list)."""
    body = _MC_ANSWER_RE.sub("", prompt)
    cut = re.search(r"\n[A-Z]\)\s", body)
    if cut:
        body = body[:cut.start()]
    for prefix in ("Context: ", "Question: "):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    return body.rstrip()


@torch.no_grad()
def _score_logits(
    model: torch.nn.Module,
    tokenizer,
    context: str,
    continuation: str,
    device: torch.device,
    context_length: int,
) -> tuple[float, int]:
    """Return (sum_logprob, n_tokens) of `continuation` given `context`. Caller normalizes."""
    ctx_ids = tokenizer.encode(context)
    cont_ids = tokenizer.encode(continuation)
    if not cont_ids:
        return float("-inf"), 0

    full_ids = ctx_ids + cont_ids
    if len(full_ids) > context_length:
        keep_ctx = max(context_length - len(cont_ids), 1)
        full_ids = ctx_ids[-keep_ctx:] + cont_ids
        cont_start = keep_ctx
    else:
        cont_start = len(ctx_ids)

    idx = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(idx)
    log_probs = logits.log_softmax(dim=-1)

    # logits[0, t] predict token at position t+1, so logP(full_ids[pos]) is
    # log_probs[0, pos - 1, full_ids[pos]]. Skip pos==0 (no prefix to score).
    total = 0.0
    n = 0
    for i in range(len(cont_ids)):
        pos = cont_start + i
        if pos == 0:
            continue
        total += log_probs[0, pos - 1, full_ids[pos]].item()
        n += 1

    return total, n


@torch.no_grad()
def score_mc_options(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    options: dict[str, str],
    device: torch.device,
    context_length: int,
    pmi: bool = False,
) -> dict[str, float]:
    """Length-normalized loglik of each option given the bare stem.

    WinoGrande-style stems with an underscore use substitution cloze
    (substitute the option into the blank, score the post-blank tail).
    Other stems use standard cloze.

    With pmi=True, subtract the option's loglik against a neutral prefix
    (Calibrate-Before-Use, Zhao et al. 2021). Off for substitution.
    """
    stem = extract_mc_stem(prompt)
    scores: dict[str, float] = {}
    is_substitution = "_" in stem
    use_pmi = pmi and not is_substitution

    if is_substitution:
        before, after = stem.split("_", 1)
        for letter, opt_text in options.items():
            opt_text = opt_text.strip()
            if not opt_text:
                scores[letter] = float("-inf")
                continue
            ctx = before + opt_text
            sum_lp, n = _score_logits(model, tokenizer, ctx, after,
                                       device, context_length)
            scores[letter] = sum_lp / n if n > 0 else float("-inf")
        return scores

    for letter, opt_text in options.items():
        opt_text = opt_text.strip()
        if not opt_text:
            scores[letter] = float("-inf")
            continue
        sum_lp, n = _score_logits(model, tokenizer, stem, " " + opt_text,
                                   device, context_length)
        score = sum_lp / n if n > 0 else float("-inf")
        if use_pmi and n > 0:
            base_lp, base_n = _score_logits(
                model, tokenizer, "Answer:", " " + opt_text,
                device, context_length,
            )
            base = base_lp / base_n if base_n > 0 else 0.0
            score = score - base
        scores[letter] = score
    return scores


def score_continuation_loglik(
    model, tokenizer, prompt, continuation, device, context_length,
):
    return _score_logits(
        model, tokenizer, prompt, " " + continuation.strip(),
        device, context_length,
    )


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
    if leaderboard:
        # LAMBADA prompts arrive with a trailing space; stripping it keeps BPE alignment clean.
        input_text = input_text.rstrip()
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
            # Cloze-scoring path: rather than greedily generating the first
            # letter token, we score each option's full text under the
            # model and emit the letter of the highest-likelihood option.
            # The runner only sees the single letter on stdout, so this is
            # a pure inference-time change — no benchmark exposure, no
            # weight updates, fully compatible with the leaderboard
            # contract. Mirrors how lm-eval-harness scores HellaSwag/OBQA
            # and recovers the per-option semantic signal that constrained
            # argmax over the letter token throws away.
            options = parse_mc_options(input_text)
            if options is not None and set(options.keys()) == set(letters):
                scores = score_mc_options(
                    model, tokenizer, input_text, options,
                    device, mc["context_length"],
                    pmi=True,
                )
                best = max(scores, key=scores.get)
                sys.stdout.write(best)
                return
            # Parsing failed — fall back to constrained-argmax greedy decode.
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
