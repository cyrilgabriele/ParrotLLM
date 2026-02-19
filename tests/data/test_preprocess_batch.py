"""Tests for batched preprocessing helpers."""

import pytest
from datasets import Dataset

from src.data.preprocess import (
    sanitize_batch,
    sanitize_document_text,
    _shingle_hashes,
    _minhash_signature_batch,
    _jaccard_from_signatures,
    deduplicate_corpus,
    MinHasher,
    DEDUP_NUM_PERM,
    DEDUP_SHINGLE_SIZE,
)


class TestDeduplication:
    """Tests for MinHash fuzzy deduplication helpers."""

    def test_shingle_hashes_basic(self):
        text = "the quick brown fox jumps over the lazy dog near the river"
        hashes = _shingle_hashes(text)
        assert len(hashes) > 0
        assert all(isinstance(h, int) for h in hashes)

    def test_shingle_hashes_short_text(self):
        text = "too few words"
        hashes = _shingle_hashes(text, size=DEDUP_SHINGLE_SIZE)
        assert hashes == set()

    def test_minhash_signature_length(self):
        texts = ["the quick brown fox jumps over the lazy dog near the river bank on a sunny day"]
        result = _minhash_signature_batch(texts)
        assert "minhash_sig" in result
        assert len(result["minhash_sig"]) == 1
        assert len(result["minhash_sig"][0]) == DEDUP_NUM_PERM

    def test_identical_docs_high_jaccard(self):
        text = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day"
        result = _minhash_signature_batch([text, text])
        jac = _jaccard_from_signatures(result["minhash_sig"][0], result["minhash_sig"][1])
        assert jac == 1.0

    def test_different_docs_low_jaccard(self):
        text_a = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day"
        text_b = "quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles"
        result = _minhash_signature_batch([text_a, text_b])
        jac = _jaccard_from_signatures(result["minhash_sig"][0], result["minhash_sig"][1])
        assert jac < 0.5

    def test_near_duplicate_detected(self):
        # Need long texts so a single word change is a small fraction of shingles
        base = (
            "the quick brown fox jumps over the lazy dog near the river bank on a sunny day "
            "in the park while the birds are singing and the children are playing with their "
            "toys in the grass and the wind is blowing gently through the trees and the flowers "
            "are blooming in the garden and the sun is shining brightly in the clear blue sky "
            "above the rolling hills and green meadows stretching far into the distance beyond "
            "the old stone wall that borders the property near the winding country road"
        )
        # Minor edit: change one word in a long document
        edited = base.replace("jumps", "leaps")
        result = _minhash_signature_batch([base, edited])
        jac = _jaccard_from_signatures(result["minhash_sig"][0], result["minhash_sig"][1])
        assert jac > 0.8

    def test_deduplicate_corpus_removes_dupes(self):
        doc_a = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day in the park"
        doc_b = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day in the park"
        doc_c = "quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles"
        texts = [doc_a, doc_b, doc_c]
        sigs = _minhash_signature_batch(texts)
        ds = Dataset.from_dict({"text": texts, "minhash_sig": sigs["minhash_sig"]})
        to_remove = deduplicate_corpus(ds)
        assert len(to_remove) == 1  # one of the two identical docs removed
        assert 2 not in to_remove  # the unique doc survives

    def test_deduplicate_keeps_longest(self):
        base = (
            "the quick brown fox jumps over the lazy dog near the river bank on a sunny day "
            "in the park while the birds are singing and the children are playing with their "
            "toys in the grass and the wind is blowing gently through the trees and the flowers "
            "are blooming in the garden and the sun is shining brightly in the clear blue sky "
            "above the rolling hills and green meadows stretching far into the distance beyond "
            "the old stone wall that borders the property near the winding country road"
        )
        short = base
        # Add only a few words so Jaccard stays above 0.8
        long = base + " with beautiful scenery all around"
        unique = (
            "quantum mechanics describes nature at the smallest scales of energy levels of atoms "
            "and subatomic particles in a framework that differs fundamentally from classical physics "
            "by introducing concepts such as wave particle duality and the uncertainty principle"
        )
        texts = [short, long, unique]
        sigs = _minhash_signature_batch(texts)
        ds = Dataset.from_dict({"text": texts, "minhash_sig": sigs["minhash_sig"]})
        to_remove = deduplicate_corpus(ds)
        # The short doc (index 0) should be removed, the long one (index 1) kept
        assert 0 in to_remove
        assert 1 not in to_remove
        assert 2 not in to_remove


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
    normalize_and_fingerprint,
    FINGERPRINT_LOWERCASE,
)


class TestDecontaminateBatch:
    @pytest.fixture
    def empty_index(self):
        return TestDecontaminationIndex(content_hashes=set())

    @pytest.fixture
    def index_with_hashes(self):
        """Index containing SHA-1 hash from a known 'test set' document."""
        test_doc = "The quick brown fox jumps over the lazy dog near the river bank"
        _, fp = normalize_and_fingerprint(test_doc)
        return TestDecontaminationIndex(
            content_hashes={fp} if fp else set(),
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

    def test_exact_hash_match_filtered(self, index_with_hashes):
        from src.data.preprocess import decontaminate_batch

        # This exact text should match the hash in the index
        texts = ["The quick brown fox jumps over the lazy dog near the river bank"]
        result = decontaminate_batch(texts, index_with_hashes)
        assert result["decontam_status"][0] == "test_overlap"

    def test_matches_sequential(self, index_with_hashes):
        from src.data.preprocess import decontaminate_batch

        texts = [
            "Completely unrelated text about programming languages and software development.",
            "The quick brown fox jumps over the lazy dog near the river bank",
            "",
        ]
        batch_result = decontaminate_batch(texts, index_with_hashes)

        for i, text in enumerate(texts):
            if not text.strip():
                assert batch_result["decontam_status"][i] == "empty_after_normalize"
                continue
            _, fp = normalize_and_fingerprint(text, lowercase=FINGERPRINT_LOWERCASE)
            if fp and fp in index_with_hashes.content_hashes:
                assert batch_result["decontam_status"][i] == "test_overlap"
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


# ── Heuristic helper unit tests ──────────────────────────────────────────────

from src.data.preprocess import (
    _has_code_fences,
    _html_tag_density,
    _symbol_to_text_ratio,
    _programming_keyword_count,
    _longest_word_length,
    _max_ngram_repeats,
    heuristic_code_filter_batch,
    heuristic_quality_filter_batch,
    HTML_TAG_DENSITY_THRESHOLD,
    CODE_SYMBOL_RATIO_THRESHOLD,
    PROGRAMMING_KEYWORD_MAX,
    MIN_WORD_COUNT,
    MIN_CHAR_COUNT,
    MAX_WORD_LENGTH,
    NGRAM_SIZE,
    NGRAM_MAX_REPEATS,
)


class TestHeuristicHelpers:
    def test_has_code_fences_true(self):
        assert _has_code_fences("Some text\n```python\ncode\n```") is True

    def test_has_code_fences_false(self):
        assert _has_code_fences("Normal prose without fences.") is False

    def test_html_tag_density_no_tags(self):
        assert _html_tag_density("Hello world") == 0.0

    def test_html_tag_density_with_tags(self):
        text = "<div>Hello</div>"
        density = _html_tag_density(text)
        # <div> = 5 chars, </div> = 6 chars → 11/16 ≈ 0.6875
        assert density > 0.5

    def test_html_tag_density_empty(self):
        assert _html_tag_density("") == 0.0

    def test_symbol_to_text_ratio_no_symbols(self):
        assert _symbol_to_text_ratio("hello world") == 0.0

    def test_symbol_to_text_ratio_with_symbols(self):
        text = "{}[]<>=\\"
        ratio = _symbol_to_text_ratio(text)
        assert ratio == 1.0

    def test_programming_keyword_count_none(self):
        assert _programming_keyword_count("The cat sat on the mat.") == 0

    def test_programming_keyword_count_some(self):
        text = "Use def to create a lambda with yield and self for async processing."
        count = _programming_keyword_count(text)
        assert count >= 3  # def, lambda, yield, self, async

    def test_longest_word_length(self):
        assert _longest_word_length("hi there") == 5
        assert _longest_word_length("") == 0
        assert _longest_word_length("superlongword") == 13

    def test_max_ngram_repeats_no_repeats(self):
        text = " ".join(f"word{i}" for i in range(20))
        assert _max_ngram_repeats(text, 10) == 1

    def test_max_ngram_repeats_with_repeats(self):
        # Repeat the same 10-word phrase 5 times
        phrase = "the quick brown fox jumps over the lazy dog now"
        text = " ".join([phrase] * 5)
        assert _max_ngram_repeats(text, 10) >= 4

    def test_max_ngram_repeats_short_text(self):
        assert _max_ngram_repeats("too short", 10) == 0


class TestHeuristicCodeFilter:
    def test_prose_kept(self, prose_text):
        result = heuristic_code_filter_batch([prose_text])
        assert result["code_filter_status"][0] == "kept"

    def test_code_fences_filtered(self):
        text = "Here is some code:\n```python\ndef foo(): pass\n```\nEnd."
        result = heuristic_code_filter_batch([text])
        assert result["code_filter_status"][0] == "code_fence"

    def test_html_density_filtered(self):
        # Make a string that's mostly HTML tags
        text = "<div><span><a href='x'>y</a></span></div>"
        result = heuristic_code_filter_batch([text])
        assert result["code_filter_status"][0] == "html_density"

    def test_symbol_ratio_filtered(self):
        # String with >10% code symbols
        text = "a{}b[]c<>d=e\\f" * 10
        result = heuristic_code_filter_batch([text])
        assert result["code_filter_status"][0] == "symbol_ratio"

    def test_keyword_density_filtered(self):
        # Use code-specific keywords that wouldn't appear in normal prose
        text = "use def to define a lambda with yield and self and await async void struct"
        count = _programming_keyword_count(text)
        assert count > PROGRAMMING_KEYWORD_MAX, f"expected >{PROGRAMMING_KEYWORD_MAX}, got {count}"
        result = heuristic_code_filter_batch([text])
        assert result["code_filter_status"][0] == "keyword_density"

    def test_boundary_symbol_ratio_kept(self):
        # Exactly at threshold should be kept (strictly >)
        # 10% symbols in 100 chars → 10 symbol chars
        filler = "a" * 90
        symbols = "{" * 10
        text = filler + symbols
        ratio = _symbol_to_text_ratio(text)
        assert ratio == 0.10
        result = heuristic_code_filter_batch([text])
        assert result["code_filter_status"][0] == "kept"

    def test_empty_batch(self):
        result = heuristic_code_filter_batch([])
        assert result == {"code_filter_status": []}

    def test_multiple_docs(self, prose_text):
        code = "```js\nconsole.log('hi');\n```"
        result = heuristic_code_filter_batch([prose_text, code])
        assert result["code_filter_status"][0] == "kept"
        assert result["code_filter_status"][1] == "code_fence"


class TestHeuristicQualityFilter:
    def test_normal_text_kept(self, prose_text):
        result = heuristic_quality_filter_batch([prose_text])
        assert result["quality_filter_status"][0] == "kept"

    def test_too_few_words(self):
        text = "Just a few words here."
        result = heuristic_quality_filter_batch([text])
        assert result["quality_filter_status"][0] == "too_few_words"

    def test_too_few_chars(self):
        # Enough words but under 200 chars: use very short words
        words = ["ok"] * 55  # 55 words but only 55*3-1=164 chars
        text = " ".join(words)
        assert len(text.split()) >= MIN_WORD_COUNT
        assert len(text) < MIN_CHAR_COUNT
        result = heuristic_quality_filter_batch([text])
        assert result["quality_filter_status"][0] == "too_few_chars"

    def test_unnatural_word(self):
        # Normal-length text with one absurdly long "word"
        words = ["normal"] * 55 + ["a" * 50]
        text = " ".join(words)
        result = heuristic_quality_filter_batch([text])
        assert result["quality_filter_status"][0] == "unnatural_word"

    def test_ngram_repetition(self):
        # Repeat a 10-word phrase many times
        phrase = "the quick brown fox jumps over the lazy dog now"
        text = " ".join([phrase] * 10)  # 100 words, repeated 10-grams
        result = heuristic_quality_filter_batch([text])
        assert result["quality_filter_status"][0] == "ngram_repetition"

    def test_boundary_exactly_50_words_kept(self):
        # Exactly 50 words, each 5 chars → 50*5 + 49 spaces = 299 chars > 200
        words = ["hello"] * 50
        text = " ".join(words)
        assert len(text.split()) == MIN_WORD_COUNT
        assert len(text) >= MIN_CHAR_COUNT
        # No unnatural words, no ngram repeats (all same word but 10-gram
        # repeats would be "hello"*10 repeated — that exceeds threshold)
        # Actually this will be caught by ngram_repetition for 10-gram of "hello"*10
        # repeated 41 times. Let's use distinct words instead.
        words = [f"word{i:03d}" for i in range(50)]
        text = " ".join(words)
        assert len(text.split()) == MIN_WORD_COUNT
        result = heuristic_quality_filter_batch([text])
        assert result["quality_filter_status"][0] == "kept"

    def test_boundary_exactly_200_chars_kept(self):
        # Need >= 50 words AND >= 200 chars
        words = [f"word{i:02d}" for i in range(55)]
        text = " ".join(words)
        assert len(text.split()) >= MIN_WORD_COUNT
        assert len(text) >= MIN_CHAR_COUNT
        result = heuristic_quality_filter_batch([text])
        assert result["quality_filter_status"][0] == "kept"

    def test_empty_batch(self):
        result = heuristic_quality_filter_batch([])
        assert result == {"quality_filter_status": []}


# Classifier tests — only run when model files exist
CODE_CLASSIFIER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "code_vs_prose.ftz"
)
HAS_CODE_CLASSIFIER = os.path.exists(CODE_CLASSIFIER_PATH)

KENLM_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "wikipedia_en.arpa.bin"
)
EDU_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "educational_quality.ftz"
)
HAS_QUALITY_MODELS = os.path.exists(KENLM_MODEL_PATH) and os.path.exists(EDU_MODEL_PATH)


@pytest.mark.skipif(not HAS_CODE_CLASSIFIER, reason="code classifier model not found")
class TestClassifierCodeFilter:
    def test_returns_dict_with_required_keys(self):
        from src.data.preprocess import classifier_code_filter_batch

        texts = ["Hello world, this is normal English prose."]
        result = classifier_code_filter_batch(texts, CODE_CLASSIFIER_PATH)
        assert "code_filter_status" in result
        assert "code_score" in result
        assert len(result["code_filter_status"]) == 1

    def test_empty_batch(self):
        from src.data.preprocess import classifier_code_filter_batch

        result = classifier_code_filter_batch([], CODE_CLASSIFIER_PATH)
        assert result == {"code_filter_status": [], "code_score": []}


@pytest.mark.skipif(not HAS_QUALITY_MODELS, reason="quality classifier models not found")
class TestClassifierQualityFilter:
    def test_returns_dict_with_required_keys(self):
        import kenlm
        from src.data.preprocess import classifier_quality_filter_batch

        kenlm_model = kenlm.Model(KENLM_MODEL_PATH)
        texts = ["This is a well-written paragraph about science and learning."]
        result = classifier_quality_filter_batch(texts, kenlm_model, EDU_MODEL_PATH)
        assert "quality_filter_status" in result
        assert "perplexity" in result
        assert "edu_score" in result
        assert len(result["quality_filter_status"]) == 1
