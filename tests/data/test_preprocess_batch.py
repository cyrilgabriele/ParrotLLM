"""Tests for batched preprocessing helpers."""

from src.data.preprocess import sanitize_batch, sanitize_document_text


class TestSanitizeBatch:
    def test_returns_dict_with_required_keys(self, sample_texts):
        result = sanitize_batch(sample_texts)
        assert "text" in result
        assert "sanitize_status" in result
        assert len(result["text"]) == len(sample_texts)
        assert len(result["sanitize_status"]) == len(sample_texts)

    def test_matches_sequential_sanitize(self, sample_texts):
        """Batch helper must produce identical results to sequential calls."""
        batch_result = sanitize_batch(sample_texts)

        for i, text in enumerate(sample_texts):
            seq_text, seq_status = sanitize_document_text(text)
            assert batch_result["text"][i] == (seq_text or "")
            assert batch_result["sanitize_status"][i] == (seq_status or "kept")

    def test_empty_batch(self):
        result = sanitize_batch([])
        assert result == {"text": [], "sanitize_status": []}

    def test_html_stripped(self):
        result = sanitize_batch(["<p>Hello <b>world</b></p> enough text here"])
        assert "<p>" not in result["text"][0]
        assert "<b>" not in result["text"][0]

    def test_code_filtered(self):
        code = (
            "```python\ndef hello():\n    print('hello')\n```\n"
            "function main() { return 0; }\nclass Foo { public void bar() {} }"
        )
        result = sanitize_batch([code])
        assert result["sanitize_status"][0] == "code_snippet"
