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


import os
import pytest

LANG_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "lid.176.ftz"
)
HAS_LANG_MODEL = os.path.exists(LANG_MODEL_PATH)


@pytest.mark.skipif(not HAS_LANG_MODEL, reason="fasttext model not found")
class TestDetectLanguageBatch:
    def test_returns_dict_with_required_keys(self):
        from src.data.preprocess import detect_language_batch

        texts = ["Hello world, this is English.", "Bonjour le monde."]
        result = detect_language_batch(texts, LANG_MODEL_PATH)
        assert "lang" in result
        assert "lang_conf" in result
        assert len(result["lang"]) == 2
        assert len(result["lang_conf"]) == 2

    def test_detects_english(self):
        from src.data.preprocess import detect_language_batch

        texts = ["The president of the United States gave a speech today."]
        result = detect_language_batch(texts, LANG_MODEL_PATH)
        assert result["lang"][0] == "en"
        assert result["lang_conf"][0] > 0.5

    def test_detects_non_english(self):
        from src.data.preprocess import detect_language_batch

        texts = ["Dies ist ein deutscher Text ueber Wissenschaft und Forschung."]
        result = detect_language_batch(texts, LANG_MODEL_PATH)
        assert result["lang"][0] == "de"

    def test_matches_sequential(self):
        from src.data.preprocess import detect_language, detect_language_batch
        import fasttext

        model = fasttext.load_model(LANG_MODEL_PATH)
        texts = [
            "Hello this is a test.",
            "Bonjour le monde entier.",
            "",
            "The quick brown fox jumps.",
        ]
        batch_result = detect_language_batch(texts, LANG_MODEL_PATH)

        for i, text in enumerate(texts):
            seq_lang, seq_conf = detect_language(text, model)
            assert batch_result["lang"][i] == seq_lang
            assert abs(batch_result["lang_conf"][i] - seq_conf) < 1e-6

    def test_empty_batch(self):
        from src.data.preprocess import detect_language_batch

        result = detect_language_batch([], LANG_MODEL_PATH)
        assert result == {"lang": [], "lang_conf": []}


from src.data.preprocess import (
    TestDecontaminationIndex,
    extract_ngrams,
    normalize_and_fingerprint,
    is_contaminated,
    FINGERPRINT_LOWERCASE,
)


class TestDecontaminateBatch:
    @pytest.fixture
    def empty_index(self):
        return TestDecontaminationIndex(
            ngrams=set(),
            content_hashes=set(),
            minhash_index=None,
            doc_shingles=None,
        )

    @pytest.fixture
    def index_with_ngrams(self):
        """Index containing n-grams from a known 'test set' document."""
        test_doc = "The quick brown fox jumps over the lazy dog near the river bank"
        ngrams = extract_ngrams(test_doc)
        _, fp = normalize_and_fingerprint(test_doc)
        return TestDecontaminationIndex(
            ngrams=ngrams,
            content_hashes={fp} if fp else set(),
            minhash_index=None,
            doc_shingles=None,
        )

    def test_returns_dict_with_required_keys(self, empty_index):
        from src.data.preprocess import decontaminate_batch

        texts = ["Hello world test document."]
        result = decontaminate_batch(texts, empty_index)
        assert "decontam_status" in result
        assert len(result["decontam_status"]) == 1

    def test_clean_docs_pass(self, empty_index):
        from src.data.preprocess import decontaminate_batch

        texts = ["This is a perfectly clean document with no overlap."]
        result = decontaminate_batch(texts, empty_index)
        assert result["decontam_status"][0] == "kept"

    def test_exact_hash_match_filtered(self, index_with_ngrams):
        from src.data.preprocess import decontaminate_batch

        # This exact text should match the hash in the index
        texts = ["The quick brown fox jumps over the lazy dog near the river bank"]
        result = decontaminate_batch(texts, index_with_ngrams)
        assert result["decontam_status"][0] == "test_overlap"

    def test_matches_sequential(self, index_with_ngrams):
        from src.data.preprocess import decontaminate_batch

        texts = [
            "Completely unrelated text about programming languages and software development.",
            "The quick brown fox jumps over the lazy dog near the river bank",
            "",
        ]
        batch_result = decontaminate_batch(texts, index_with_ngrams)

        for i, text in enumerate(texts):
            if not text.strip():
                assert batch_result["decontam_status"][i] == "empty_after_normalize"
                continue
            _, fp = normalize_and_fingerprint(text, lowercase=FINGERPRINT_LOWERCASE)
            if fp and fp in index_with_ngrams.content_hashes:
                assert batch_result["decontam_status"][i] == "test_overlap"
            elif index_with_ngrams.ngrams and is_contaminated(text, index_with_ngrams.ngrams):
                assert batch_result["decontam_status"][i] == "contaminated"
            else:
                assert batch_result["decontam_status"][i] == "kept"

    def test_empty_batch(self, empty_index):
        from src.data.preprocess import decontaminate_batch

        result = decontaminate_batch([], empty_index)
        assert result == {"decontam_status": []}


from transformers import GPT2TokenizerFast


class TestTokenizeBatch:
    @pytest.fixture
    def tokenizer(self):
        return GPT2TokenizerFast.from_pretrained("gpt2")

    def test_returns_dict_with_required_keys(self, tokenizer):
        from src.data.preprocess import tokenize_batch

        texts = ["Hello world, this is a test."]
        result = tokenize_batch(texts, tokenizer)
        assert "input_ids" in result
        assert "n_tokens" in result
        assert len(result["input_ids"]) == 1
        assert len(result["n_tokens"]) == 1

    def test_matches_sequential(self, tokenizer, english_text):
        from src.data.preprocess import tokenize_batch

        texts = [english_text, "Short.", "Another longer sentence with more words."]
        batch_result = tokenize_batch(texts, tokenizer)

        for i, text in enumerate(texts):
            seq_tokens = tokenizer.encode(text)
            assert batch_result["input_ids"][i] == seq_tokens
            assert batch_result["n_tokens"][i] == len(seq_tokens)

    def test_empty_batch(self, tokenizer):
        from src.data.preprocess import tokenize_batch

        result = tokenize_batch([], tokenizer)
        assert result == {"input_ids": [], "n_tokens": []}

    def test_token_count_correct(self, tokenizer):
        from src.data.preprocess import tokenize_batch

        texts = ["Hello world"]
        result = tokenize_batch(texts, tokenizer)
        assert result["n_tokens"][0] == len(result["input_ids"][0])
