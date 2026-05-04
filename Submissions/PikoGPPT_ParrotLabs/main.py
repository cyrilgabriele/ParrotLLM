#!/usr/bin/env python3
"""Minimal leaderboard submission entrypoint for ParrotLLM."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from src.inference import (
    cloze_score_options,
    detect_mc_prompt,
    is_lambada_shape,
    letter_token_ids,
    score_continuation_logprob,
    wino_substitute,
)
from src.model.transformer import ParrotLLM


DEFAULT_TOKENIZER_NAME = "openai-community/gpt2"

# Must match the system_prompt used during SFT/DPO training
# (configs/posttraining/*.yaml). Course rule (VL07 slide 32):
# the template at training MUST match the template at inference.
DEFAULT_SYSTEM_PROMPT = "You are ParrotLLM, a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParrotLLM leaderboard submission")
    parser.add_argument("--stage", required=True, choices=["inference"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--template",
        choices=["alpaca", "raw"],
        default="alpaca",
        help="alpaca: wrap prompt in ### Instruction/### Response (matches SFT/DPO training). "
        "raw: pass prompt unchanged (use only for base-model checkpoints).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended to user content under alpaca template. "
        "Must match the value used during training.",
    )
    return parser.parse_args()


def alpaca_wrap(user_prompt: str, system_prompt: str) -> str:
    """Wrap a raw prompt in the Alpaca template used at training time.

    Mirrors src/posttraining/templates.py:render_conversation(template_format='alpaca')
    with add_generation_prompt=True. The system prompt is folded into the first
    user message exactly as normalize_messages() does it.
    """
    sys = (system_prompt or "").strip()
    user = user_prompt if user_prompt is not None else ""
    if sys:
        user = f"{sys}\n\n{user}"
    return f"### Instruction:\n{user}\n\n### Response:\n"


@dataclass
class RenderedPrompt:
    kind: str  # "mc" | "lambada" | "chat"
    text: str
    mc_options: list[str] = field(default_factory=list)
    mc_header: str = ""
    mc_stem: str = ""


def render_prompt_for_inference(
    *,
    raw_prompt: str,
    template: str,
    system_prompt: str,
    leaderboard: bool,
) -> RenderedPrompt:
    """Decide which inference path to take for `raw_prompt`.

    In leaderboard mode the leaderboard sends raw benchmark prompts and reads
    STDOUT directly, so the Alpaca wrapper MUST NOT be applied to MC/LAMBADA
    inputs (it would destroy the surface form). Only chat-shaped prompts (or
    non-leaderboard runs) get wrapped.
    """
    if leaderboard:
        mc = detect_mc_prompt(raw_prompt)
        if mc is not None:
            stem, options, header = mc
            return RenderedPrompt(
                kind="mc",
                text=raw_prompt,
                mc_options=options,
                mc_header=header,
                mc_stem=stem,
            )
        if is_lambada_shape(raw_prompt):
            return RenderedPrompt(kind="lambada", text=raw_prompt.rstrip())

    if template == "alpaca":
        return RenderedPrompt(kind="chat", text=alpaca_wrap(raw_prompt, system_prompt))
    return RenderedPrompt(kind="chat", text=raw_prompt)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(submission_dir: Path) -> GPT2TokenizerFast:
    local_candidates = [
        submission_dir / "tokenizer",
        submission_dir / "src" / "tokenizer",
        submission_dir / "assets" / "tokenizer",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return GPT2TokenizerFast.from_pretrained(candidate, use_fast=True)
    return GPT2TokenizerFast.from_pretrained(DEFAULT_TOKENIZER_NAME, use_fast=True)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ParrotLLM, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict) or "model" not in raw_config:
        raise ValueError("Checkpoint must contain checkpoint['config']['model'].")

    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain checkpoint['model'] state dict.")

    config = {"model": raw_config["model"]}
    model = ParrotLLM(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    context_length: int,
    eos_token_id: int | None = None,
    allowed_first_token_ids: list[int] | None = None,
) -> torch.Tensor:
    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if step == 0 and allowed_first_token_ids:
            vocab_size = logits.size(-1)
            in_range = [tid for tid in allowed_first_token_ids if 0 <= tid < vocab_size]
            if in_range:
                mask = torch.full_like(logits, float("-inf"))
                mask[:, in_range] = 0.0
                logits = logits + mask

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = sorted_logits.softmax(dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = cumulative_probs - sorted_probs > top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return idx


def _score_mc(
    *,
    model,
    tokenizer,
    rendered: RenderedPrompt,
    device,
    context_length: int,
) -> int:
    """Run cloze scoring for an MC prompt and return the chosen option index."""
    n_opts = len(rendered.mc_options)
    if rendered.mc_header == "Context" and n_opts == 2 and "_" in rendered.mc_stem:
        # WinoGrande: substitute the option into the blank, score the post-blank tail.
        best_idx, best_score = 0, float("-inf")
        for i, opt in enumerate(rendered.mc_options):
            head, tail = wino_substitute(rendered.mc_stem, opt)
            head_ids = tokenizer.encode(head, add_special_tokens=False)
            tail_ids = tokenizer.encode(tail, add_special_tokens=False) if tail else []
            if not tail_ids:
                # Degenerate: score the option itself given the prefix-only context.
                prefix = rendered.mc_stem.split("_")[0].rstrip()
                score = score_continuation_logprob(
                    model,
                    prefix_ids=tokenizer.encode(prefix, add_special_tokens=False),
                    continuation_ids=tokenizer.encode(
                        " " + opt, add_special_tokens=False
                    ),
                    device=device,
                    context_length=context_length,
                )
            else:
                score = score_continuation_logprob(
                    model,
                    prefix_ids=head_ids,
                    continuation_ids=tail_ids,
                    device=device,
                    context_length=context_length,
                )
            if score > best_score:
                best_score, best_idx = score, i
        return best_idx
    # HellaSwag / OpenBookQA: classic cloze on the full prompt + " <option>".
    return cloze_score_options(
        model,
        tokenizer,
        prefix_text=rendered.text,
        option_texts=rendered.mc_options,
        device=device,
        context_length=context_length,
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    submission_dir = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device(args.device)
    model, config = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(submission_dir)

    raw_prompt = args.prompt if args.prompt is not None else "The answer is"
    rendered = render_prompt_for_inference(
        raw_prompt=raw_prompt,
        template=args.template,
        system_prompt=args.system_prompt,
        leaderboard=bool(args.leaderboard),
    )

    context_length = int(config["model"].get("context_length", 1024))
    eos_id = config["model"].get("eos_token_id")
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    # ── MC path: cloze-score the options, write the chosen letter and exit ──
    if rendered.kind == "mc":
        try:
            best_idx = _score_mc(
                model=model,
                tokenizer=tokenizer,
                rendered=rendered,
                device=device,
                context_length=context_length,
            )
            letter = chr(ord("A") + best_idx)
        except Exception:
            # Degraded fallback: constrain the first generated token to one of
            # the allowed letters. Guarantees we emit *some* valid letter even
            # if cloze scoring trips on a malformed input.
            n_opts = max(2, len(rendered.mc_options))
            allowed = letter_token_ids(
                tokenizer, [chr(ord("A") + k) for k in range(n_opts)]
            )
            input_ids = tokenizer.encode(rendered.text, add_special_tokens=False)
            if not input_ids:
                input_ids = [int(eos_id) if eos_id is not None else 0]
            idx_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            out = generate(
                model,
                idx_t,
                max_new_tokens=1,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                context_length=context_length,
                allowed_first_token_ids=allowed,
            )
            generated = tokenizer.decode(
                out[0, len(input_ids):].tolist(),
                clean_up_tokenization_spaces=False,
            ).lstrip()
            letter = generated[:1].upper() if generated else "A"
        if args.leaderboard:
            sys.stdout.write(letter)
            sys.stdout.flush()
            return
        print(letter)
        return

    # ── LAMBADA / chat path: greedy-generate, stop on EOS ──
    input_ids = tokenizer.encode(rendered.text, add_special_tokens=False)
    if not input_ids:
        if eos_id is None:
            raise ValueError("Prompt is empty and tokenizer has no EOS token.")
        input_ids = [eos_id]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = generate(
        model,
        idx,
        max_new_tokens=max(0, int(args.max_tokens)),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        context_length=context_length,
        eos_token_id=int(eos_id) if eos_id is not None else None,
    )

    generated_ids = output[0, len(input_ids):].tolist()
    # Trim a trailing EOS so it doesn't pollute the leaderboard's parsing.
    if eos_id is not None and generated_ids and generated_ids[-1] == eos_id:
        generated_ids = generated_ids[:-1]
    generated_text = tokenizer.decode(
        generated_ids,
        clean_up_tokenization_spaces=False,
    )

    if args.leaderboard:
        sys.stdout.write(generated_text)
        sys.stdout.flush()
        return

    full_text = tokenizer.decode(output[0].tolist(), clean_up_tokenization_spaces=False)
    print(full_text)


if __name__ == "__main__":
    main()
