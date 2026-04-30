"""System prompt must appear in rendered training text (Alpaca-merge style).

Stanford Alpaca's PROMPT_DICT prepends 'Below is an instruction...' into
the user-facing instruction. We do the equivalent: prepend the YAML
system_prompt onto the first user turn so the model is supervised on it.
"""
from src.posttraining.templates import normalize_messages, render_conversation


def test_system_prompt_merged_into_first_user_turn():
    messages = [
        {"role": "system", "content": "You are ParrotLLM, a helpful assistant."},
        {"role": "user", "content": "Hello there."},
        {"role": "assistant", "content": "Hi!"},
    ]
    normalized = normalize_messages(
        messages,
        system_prompt="You are ParrotLLM, a helpful assistant.",
    )
    rendered = render_conversation(normalized)
    assert "You are ParrotLLM, a helpful assistant." in rendered.text, (
        "system_prompt was dropped from rendered training text"
    )
    assert "Hello there." in rendered.text
    assert "Hi!" in rendered.text


def test_system_prompt_only_appears_once_even_for_multiturn():
    messages = [
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "A1."},
        {"role": "user", "content": "Second."},
        {"role": "assistant", "content": "A2."},
    ]
    normalized = normalize_messages(messages, system_prompt="SYS_PROMPT_TOKEN")
    rendered = render_conversation(normalized)
    assert rendered.text.count("SYS_PROMPT_TOKEN") == 1, (
        "system prompt should be merged into first user turn only, not repeated"
    )


def test_system_prompt_not_in_assistant_loss_span():
    """The merged system text lives in the user turn — must NOT be supervised."""
    messages = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi."},
    ]
    normalized = normalize_messages(messages, system_prompt="SYSTEM_TEXT")
    rendered = render_conversation(normalized)
    for span_start, span_end in rendered.assistant_spans:
        span_text = rendered.text[span_start:span_end]
        assert "SYSTEM_TEXT" not in span_text, (
            "system prompt leaked into assistant span — would corrupt loss"
        )


def test_no_system_prompt_when_none_given():
    messages = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi."},
    ]
    normalized = normalize_messages(messages, system_prompt=None)
    rendered = render_conversation(normalized)
    # Sanity: rendering still works; no spurious merge happens.
    assert "Hello." in rendered.text
