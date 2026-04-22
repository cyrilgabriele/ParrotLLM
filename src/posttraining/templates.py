"""Shared plain-text chat template used for SFT and interactive chat."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "chatbot": "assistant",
    "system": "system",
    "instruction": "system",
    "user": "user",
    "human": "user",
    "prompter": "user",
}

ROLE_HEADERS = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}

HEADER_STOP_RE = re.compile(r"(?:\n\n|\A)### (?:System|User|Assistant):")


@dataclass(slots=True)
class RenderedConversation:
    text: str
    assistant_spans: list[tuple[int, int]]
    messages: list[dict[str, str]]


@dataclass(slots=True)
class TokenizedConversation:
    tokens: list[int]
    token_loss_mask: list[int]
    text: str
    messages: list[dict[str, str]]


def canonicalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    return ROLE_ALIASES.get(str(role).strip().lower())


def clean_message_content(text: str | None) -> str:
    if text is None:
        return ""
    content = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"\bAs ChatGPT\b[:,]?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(
        r"\bAs an AI language model\b[:,]?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    return content.strip()


def normalize_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    system_prompt: str | None = None,
    require_final_assistant: bool = True,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for raw in messages:
        role = canonicalize_role(raw.get("role")) if isinstance(raw, Mapping) else None
        if role is None:
            continue
        content = clean_message_content(raw.get("content")) if isinstance(raw, Mapping) else ""
        if not content:
            continue
        if role == "system":
            if normalized and normalized[0]["role"] == "system":
                normalized[0] = {"role": "system", "content": content}
            elif not normalized:
                normalized.append({"role": "system", "content": content})
            continue
        if not normalized and role == "assistant":
            continue
        if normalized and normalized[-1]["role"] == role:
            if normalized[-1]["content"] != content:
                normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{content}".strip()
            continue
        normalized.append({"role": role, "content": content})

    if system_prompt:
        system_prompt = clean_message_content(system_prompt)
        if system_prompt:
            if normalized and normalized[0]["role"] == "system":
                normalized[0] = {"role": "system", "content": normalized[0]["content"]}
            else:
                normalized.insert(0, {"role": "system", "content": system_prompt})

    if normalized and normalized[0]["role"] == "assistant":
        normalized = normalized[1:]

    cleaned: list[dict[str, str]] = []
    prev_role: str | None = None
    for message in normalized:
        role = message["role"]
        if role == "system" and cleaned:
            continue
        if prev_role == role and role != "system":
            continue
        cleaned.append(message)
        prev_role = role

    if require_final_assistant:
        while cleaned and cleaned[-1]["role"] != "assistant":
            cleaned.pop()

    return cleaned


def render_conversation(
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> RenderedConversation:
    text_parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0
    normalized = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]

    for idx, message in enumerate(normalized):
        if idx > 0:
            text_parts.append("\n\n")
            cursor += 2
        prefix = ROLE_HEADERS[message["role"]]
        content = clean_message_content(message["content"])
        text_parts.append(prefix)
        cursor += len(prefix)
        start = cursor
        text_parts.append(content)
        cursor += len(content)
        if message["role"] == "assistant":
            assistant_spans.append((start, cursor))

    if add_generation_prompt:
        if text_parts:
            text_parts.append("\n\n")
        text_parts.append(ROLE_HEADERS["assistant"])

    return RenderedConversation(
        text="".join(text_parts),
        assistant_spans=assistant_spans,
        messages=normalized,
    )


def tokenize_conversation(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    system_prompt: str | None = None,
    add_generation_prompt: bool = False,
    append_eos: bool = False,
) -> TokenizedConversation:
    normalized = normalize_messages(
        messages,
        system_prompt=system_prompt,
        require_final_assistant=not add_generation_prompt,
    )
    rendered = render_conversation(normalized, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(
        rendered.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    tokens = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    mask: list[int] = []
    for start, end in offsets:
        active = 0
        if end > start:
            for span_start, span_end in rendered.assistant_spans:
                if start < span_end and end > span_start:
                    active = 1
                    break
        mask.append(active)
    if append_eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id when append_eos=True.")
        tokens.append(int(eos_id))
        mask.append(0)
    return TokenizedConversation(
        tokens=tokens,
        token_loss_mask=mask,
        text=rendered.text,
        messages=normalized,
    )


def trim_messages_to_token_limit(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    system_prompt: str | None,
    max_tokens: int,
    append_eos: bool = True,
) -> TokenizedConversation | None:
    working = normalize_messages(messages, system_prompt=system_prompt)
    while working:
        tokenized = tokenize_conversation(
            tokenizer,
            working,
            system_prompt=None,
            append_eos=append_eos,
        )
        if len(tokenized.tokens) <= max_tokens:
            return tokenized
        if len(working) <= 2:
            return None
        start = 1 if working and working[0]["role"] == "system" else 0
        del_count = 2 if len(working) - start >= 2 else 1
        del working[start : start + del_count]
        while working and working[-1]["role"] != "assistant":
            working.pop()
    return None


def build_generation_prompt(
    messages: Sequence[Mapping[str, str]],
    *,
    system_prompt: str | None = None,
) -> str:
    normalized = normalize_messages(
        messages,
        system_prompt=system_prompt,
        require_final_assistant=False,
    )
    return render_conversation(normalized, add_generation_prompt=True).text


def strip_generated_assistant_text(text: str) -> str:
    stripped = text.strip()
    match = HEADER_STOP_RE.search(stripped)
    if match and match.start() > 0:
        stripped = stripped[: match.start()]
    for stop in ("\n\n### User:", "\n\n### System:", "\n\n### Assistant:"):
        if stop in stripped:
            stripped = stripped.split(stop, 1)[0]
    return stripped.strip()
