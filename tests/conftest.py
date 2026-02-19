"""Shared fixtures for preprocessing tests."""

import pytest


@pytest.fixture
def sample_texts():
    """Small set of documents covering various filter cases."""
    return [
        # Normal English text - should be kept
        "The quick brown fox jumps over the lazy dog. This is a normal English "
        "document with enough tokens to pass the minimum length filter. It contains "
        "several sentences of perfectly readable prose that should survive all "
        "preprocessing stages without any issues whatsoever. We need enough text "
        "here to exceed the 64-token minimum after GPT-2 tokenization.",

        # Empty - should be filtered
        "",

        # Whitespace only - should be filtered
        "   \n\n  \t  ",

        # Code snippet - should be filtered by sanitization
        "```python\ndef hello():\n    print('hello world')\n```\n"
        "function main() { return 0; }\nclass Foo { public void bar() {} }",

        # HTML content - should be cleaned
        "<p>This is a <b>bold</b> paragraph.</p> And this is normal text that "
        "continues after the HTML tags have been stripped away. We need enough "
        "content here to survive the minimum token length filter after cleaning "
        "and tokenization by the GPT-2 tokenizer.",

        # Short text - should be filtered by min_tokens
        "Too short.",

        # Non-English (German) - should be filtered by language detection
        "Dies ist ein deutscher Text. Er sollte vom Sprachfilter erkannt und "
        "entfernt werden, da wir nur englische Dokumente behalten wollen.",
    ]


@pytest.fixture
def english_text():
    """A single clean English document guaranteed to pass all filters."""
    return (
        "The history of natural language processing generally started in the 1950s, "
        "although work can be found from earlier periods. In 1950, Alan Turing published "
        "an article titled Computing Machinery and Intelligence which proposed what is now "
        "called the Turing test as a criterion of intelligence, a task that involves the "
        "automated interpretation and generation of natural language. The premise of the "
        "test is that if the computer can engage in a conversation with a human without "
        "being detected as a machine, it has demonstrated human intelligence."
    )
