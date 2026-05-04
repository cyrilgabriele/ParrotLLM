"""Pure helpers for leaderboard inference: prompt-shape detection and parsing.

These functions are imported by main.py to dispatch each incoming prompt to the
right inference path (cloze-scored MC vs. raw next-word vs. chat). The pure
detection helpers (detect_mc_prompt, is_lambada_shape, wino_substitute) have no
torch dependency. The model-dependent helpers (cloze_score_options,
score_continuation_logprob, letter_token_ids) live in this same module so
main.py has a single import surface.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F


_OPTION_LINE_RE = re.compile(r"^([A-Z])\)\s*(.*)$")


def detect_mc_prompt(prompt: str) -> tuple[str, list[str], str] | None:
    """Detect leaderboard MC shape and parse out (stem, options, header).

    Returns None for prompts that aren't MC. A prompt qualifies as MC iff:
      - the LAST non-empty line is exactly "Answer:" (case-sensitive), and
      - it contains at least two lines matching "^[A-Z]) <text>", and
      - the line(s) above the first option line form a non-empty stem.

    The returned `header` is "Context", "Question", "Passage", or "" (no
    recognized prefix). The `stem` has the header prefix stripped (so
    "Context: foo" → stem="foo").
    """
    if "Answer:" not in prompt:
        return None
    lines = prompt.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[-1].rstrip() != "Answer:":
        return None

    option_lines: list[tuple[int, str, str]] = []  # (idx, letter, text)
    for i, line in enumerate(lines[:-1]):
        m = _OPTION_LINE_RE.match(line)
        if m:
            option_lines.append((i, m.group(1), m.group(2).strip()))
    if len(option_lines) < 2:
        return None

    expected = [chr(ord("A") + k) for k in range(len(option_lines))]
    if [letter for _, letter, _ in option_lines] != expected:
        return None

    first_opt_idx = option_lines[0][0]
    stem_lines = [line for line in lines[:first_opt_idx] if line.strip()]
    if not stem_lines:
        return None

    stem_block = "\n".join(stem_lines).strip()
    header = ""
    for candidate in ("Context", "Question", "Passage"):
        prefix = f"{candidate}:"
        if stem_block.startswith(prefix):
            header = candidate
            stem_block = stem_block[len(prefix):].lstrip()
            break

    options = [text for _, _, text in option_lines]
    if any(not opt for opt in options):
        return None

    return stem_block, options, header


def is_lambada_shape(prompt: str) -> bool:
    """LAMBADA test prompts: raw narrative passage that ends with a trailing space.

    Reject anything that already looks like a chat instruction or MC prompt.
    Require ≥80 chars to filter accidental matches like "Tell me ".
    """
    if not prompt:
        return False
    if detect_mc_prompt(prompt) is not None:
        return False
    if not prompt.endswith(" "):
        return False
    return len(prompt.strip()) >= 80


def letter_token_ids(tokenizer, letters: list[str]) -> list[int]:
    """Return token ids whose decoded form starts with one of `letters`.

    Iterates the full vocab once. Used to mask the first-step logits so the
    model can only emit a token that begins with an allowed answer letter
    (after optional leading whitespace). Stable across runs for a fixed vocab.
    """
    allowed = {ch.upper() for ch in letters}
    vocab_size = tokenizer.vocab_size
    matched: list[int] = []
    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        stripped = decoded.lstrip()
        if not stripped:
            continue
        if stripped[0].upper() in allowed:
            matched.append(tid)
    return matched


def wino_substitute(stem: str, option: str) -> tuple[str, str]:
    """Substitute `option` into the first `_` in `stem`. Return (head, tail).

    head is the prefix up to and including the option (no trailing space).
    tail is the remainder of the stem after the blank.

    If no blank is present, fall back to ("<stem> <option>", "") so the caller
    can still score under "head" without crashing.
    """
    if "_" not in stem:
        return f"{stem.rstrip()} {option}", ""
    idx = stem.index("_")
    head = stem[:idx].rstrip()
    head = f"{head} {option}" if head else option
    tail = stem[idx + 1 :]
    return head, tail
