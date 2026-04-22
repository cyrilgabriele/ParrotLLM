from __future__ import annotations

from src.posttraining.templates import (
    build_generation_prompt,
    strip_generated_assistant_text,
    tokenize_conversation,
    trim_messages_to_token_limit,
)


class DummyTokenizer:
    eos_token_id = 999

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        verbose: bool = True,
    ):
        del add_special_tokens, verbose
        payload = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload


def test_build_generation_prompt_uses_frozen_template():
    prompt = build_generation_prompt(
        [{"role": "user", "content": "Say hello."}],
        system_prompt="You are helpful.",
    )
    assert prompt.startswith("### System:\nYou are helpful.")
    assert "### User:\nSay hello." in prompt
    assert prompt.endswith("### Assistant:\n")


def test_tokenize_conversation_masks_only_assistant_content():
    tokenizer = DummyTokenizer()
    tokenized = tokenize_conversation(
        tokenizer,
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ],
        system_prompt="System",
        append_eos=True,
    )
    assert len(tokenized.tokens) == len(tokenized.token_loss_mask)
    assert sum(tokenized.token_loss_mask) == len("World")
    assert tokenized.token_loss_mask[-1] == 0


def test_trim_messages_to_token_limit_drops_earliest_turns():
    tokenizer = DummyTokenizer()
    trimmed = trim_messages_to_token_limit(
        tokenizer,
        [
            {"role": "user", "content": "A" * 20},
            {"role": "assistant", "content": "B" * 20},
            {"role": "user", "content": "Short"},
            {"role": "assistant", "content": "Answer"},
        ],
        system_prompt="System",
        max_tokens=80,
        append_eos=True,
    )
    assert trimmed is not None
    assert "Short" in trimmed.text
    assert "A" * 20 not in trimmed.text


def test_strip_generated_assistant_text_stops_at_next_header():
    text = "Sure.\n\n### User:\nAnother question"
    assert strip_generated_assistant_text(text) == "Sure."
