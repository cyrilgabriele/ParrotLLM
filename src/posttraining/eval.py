"""Small prompt-suite evaluation for SFT checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.eval.inference import generate
from .templates import build_generation_prompt, strip_generated_assistant_text


def _load_prompt_suite(path: str | Path) -> list[dict[str, Any]]:
    suite_path = Path(path)
    if not suite_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with suite_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _score_case(case: dict[str, Any], response: str) -> float:
    score = 1.0
    fmt = str(case.get("expected_format", "freeform")).strip().lower()
    normalized = response.strip()

    if fmt == "json":
        try:
            json.loads(normalized)
        except json.JSONDecodeError:
            return 0.0
    elif fmt == "mcq_letter":
        if normalized not in {"A", "B", "C", "D"}:
            return 0.0
    elif fmt == "short_answer":
        if len(normalized.split()) > int(case.get("max_words", 12)):
            score *= 0.0
    elif fmt == "refusal":
        refusal_markers = ("cannot help", "can't help", "won't help", "cannot assist", "can't assist")
        if not any(marker in normalized.lower() for marker in refusal_markers):
            score *= 0.0

    gold = case.get("gold")
    if gold is not None and normalized != str(gold).strip():
        score *= 0.0

    must_contain = case.get("must_contain") or []
    for needle in must_contain:
        if str(needle) not in normalized:
            score *= 0.0

    return score


@torch.no_grad()
def evaluate_prompt_suite(
    model: torch.nn.Module,
    tokenizer,
    *,
    path: str | Path | None,
    device: torch.device,
    system_prompt: str | None,
    context_length: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> dict[str, float]:
    if path is None:
        return {"count": 0.0, "format_score": 0.0}

    cases = _load_prompt_suite(path)
    if not cases:
        return {"count": 0.0, "format_score": 0.0}

    total = 0
    correct = 0.0
    for case in cases:
        messages = case.get("messages") or []
        if not messages:
            prompt = str(case.get("prompt", "")).strip()
            if not prompt:
                continue
            messages = [{"role": "user", "content": prompt}]
        prompt_text = build_generation_prompt(messages, system_prompt=system_prompt)
        input_ids = tokenizer.encode(prompt_text)
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = generate(
            model,
            idx,
            int(case.get("max_tokens", 64)),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            context_length=context_length,
        )
        generated = tokenizer.decode(output[0, len(input_ids):].tolist())
        response = strip_generated_assistant_text(generated)
        correct += _score_case(case, response)
        total += 1

    if total == 0:
        return {"count": 0.0, "format_score": 0.0}
    return {"count": float(total), "format_score": correct / total}
