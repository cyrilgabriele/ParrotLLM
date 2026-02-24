from __future__ import annotations

"""Data preprocessing pipeline: filter, tokenize, and save as binary splits."""

import hashlib
import html
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fasttext
import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import GPT2TokenizerFast


FINGERPRINT_LOWERCASE = True
_BASE_BOILERPLATE_EXACT = (
    "<<< begin of text >>>",
    "<<< end of text >>>",
    "*** start of this project gutenberg ebook",
    "*** end of this project gutenberg ebook",
    "*** start of the project gutenberg ebook",
    "*** end of the project gutenberg ebook",
)
_BASE_BOILERPLATE_PREFIXES = (
    "*** start of",
    "*** end of",
    "article url:",
    "source:",
    "category:",
    "tags:",
    "title:",
    "url:",
)
BOILERPLATE_EXACT_MARKERS = tuple(marker.lower() for marker in _BASE_BOILERPLATE_EXACT)
BOILERPLATE_PREFIX_MARKERS = tuple(prefix.lower() for prefix in _BASE_BOILERPLATE_PREFIXES)
WHITESPACE_RE = re.compile(r"\s+")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]*>")
URL_RE = re.compile(r"https?://\S+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Control chars, BOM, replacement char — but preserve \t and \n (handled separately)
_CONTROL_RE = re.compile(r"[\ufffd\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
CODE_KEYWORD_RE = re.compile(
    r"\b(def|function|public|private|package|namespace|var|let|const|struct|enum)\b"
)
CODE_FENCE_RE = re.compile(r"```|~~~|<code>|</code>|&lt;/?code&gt;", re.IGNORECASE)
CODE_STRUCTURAL_RE = re.compile(
    r"^\s*[\[\{]|[\]\}]\s*[,;]?\s*$"  # lines that are mostly JSON/config brackets
)

# Corpus deduplication (MinHash LSH)
DEDUP_NUM_PERM = 64
DEDUP_BANDS = 16        # 16 bands x 4 rows = 64 perms
DEDUP_ROWS = 4
DEDUP_SHINGLE_SIZE = 5  # 5-word shingles
DEDUP_THRESHOLD = 0.8   # Jaccard similarity threshold
_DEDUP_PRIME = (1 << 61) - 1  # Mersenne prime for universal hashing
_DEDUP_MAX_HASH = (1 << 32) - 1

# ── Heuristic filter thresholds (Phase 3 + 4) ────────────────────────────────
HTML_TAG_DENSITY_THRESHOLD = 0.02        # Phase 3a: >2 % HTML tag chars → drop
CODE_SYMBOL_CHARS = frozenset("{}[]<>=\\_")
CODE_SYMBOL_RATIO_THRESHOLD = 0.10       # Phase 3b: strictly >10 % → drop
PROGRAMMING_KEYWORDS = frozenset({
    # Python (code-specific only — excludes from, as, with, pass, class, import, return)
    "def", "elif", "except", "finally", "lambda", "self", "yield", "assert",
    # JavaScript / TypeScript
    "const", "async", "await", "typeof", "instanceof",
    "undefined", "prototype", "require",
    # Java / C# / C++
    "void", "boolean", "namespace", "implements", "extends",
    "override", "abstract", "virtual", "protected",
    # General (code-specific only — excludes new, case, break, continue, delete, catch, throw)
    "struct", "enum", "sizeof", "typedef", "goto",
    "template", "inline", "volatile", "extern", "register",
    "println", "printf", "malloc", "nullptr", "stdin", "stdout",
    "argv", "argc", "endl", "iostream", "strcmp",
})
PROGRAMMING_KEYWORD_MAX = 3              # Phase 3c: >3 distinct → drop

MIN_WORD_COUNT = 50                      # Phase 4a
MIN_CHAR_COUNT = 200                     # Phase 4a
MAX_WORD_LENGTH = 40                     # Phase 4b
NGRAM_SIZE = 10                          # Phase 4c
NGRAM_MAX_REPEATS = 3                    # Phase 4c: any 10-gram >3 times → drop

# ── Classifier model paths ────────────────────────────────────────────────────
CODE_CLASSIFIER_FILENAME = "code_vs_prose.ftz"
QUALITY_CLASSIFIER_FILENAME = "educational_quality.ftz"
KENLM_MODEL_FILENAME = "wikipedia_en.arpa.bin"
CODE_CLASSIFIER_THRESHOLD = 0.5
KENLM_PERPLEXITY_LOW = 10.0
KENLM_PERPLEXITY_HIGH = 100000.0
EDUCATIONAL_QUALITY_MIN = 2


class MinHasher:
    """Compute MinHash signatures using universal hashing: h(x) = (a*x + b) mod p."""

    def __init__(self, num_perm: int = DEDUP_NUM_PERM, seed: int = 42):
        import random
        rng = random.Random(seed)
        self.num_perm = num_perm
        self._a = [rng.randint(1, _DEDUP_PRIME - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, _DEDUP_PRIME - 1) for _ in range(num_perm)]

    def signature(self, hashes: set[int]) -> list[int]:
        """Compute MinHash signature from a set of shingle hashes."""
        if not hashes:
            return [_DEDUP_MAX_HASH] * self.num_perm
        sig = []
        for a, b in zip(self._a, self._b):
            min_val = min((a * h + b) % _DEDUP_PRIME for h in hashes)
            sig.append(min_val)
        return sig


class UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self):
        self.parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self._rank[x] = 0
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def _shingle_hashes(text: str, size: int = DEDUP_SHINGLE_SIZE) -> set[int]:
    """Normalize text, split into words, extract word-level shingles, hash each."""
    words = text.lower().split()
    if len(words) < size:
        return set()
    hashes = set()
    for i in range(len(words) - size + 1):
        shingle = " ".join(words[i : i + size])
        h = int(hashlib.blake2b(shingle.encode(), digest_size=8).hexdigest(), 16)
        hashes.add(h % _DEDUP_PRIME)
    return hashes


# Module-level hasher for use by batched map function
_GLOBAL_MINHASHER = MinHasher(DEDUP_NUM_PERM)


def _minhash_signature_batch(texts: list[str]) -> dict[str, list]:
    """Batch function for Dataset.map — compute MinHash signature per document."""
    sigs = []
    for text in texts:
        h = _shingle_hashes(text)
        sigs.append(_GLOBAL_MINHASHER.signature(h))
    return {"minhash_sig": sigs}


def _jaccard_from_signatures(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def deduplicate_corpus(ds) -> set[int]:
    """Build LSH index, find near-duplicate clusters, return row indices to remove.

    Keeps the longest document (by word count) in each cluster.
    """
    n = len(ds)
    # Build LSH buckets: band_idx -> band_hash -> list of doc indices
    buckets: dict[int, dict[int, list[int]]] = {
        b: {} for b in range(DEDUP_BANDS)
    }
    sigs = ds["minhash_sig"]

    for doc_idx in range(n):
        sig = sigs[doc_idx]
        for band_idx in range(DEDUP_BANDS):
            start = band_idx * DEDUP_ROWS
            band = tuple(sig[start : start + DEDUP_ROWS])
            band_hash = hash(band)
            buckets[band_idx].setdefault(band_hash, []).append(doc_idx)

    # Find candidate pairs and verify Jaccard
    uf = UnionFind()
    for band_idx in range(DEDUP_BANDS):
        for bucket in buckets[band_idx].values():
            if len(bucket) < 2:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    if uf.find(a) == uf.find(b):
                        continue  # already in same cluster
                    jac = _jaccard_from_signatures(sigs[a], sigs[b])
                    if jac >= DEDUP_THRESHOLD:
                        uf.union(a, b)

    # Group clusters
    clusters: dict[int, list[int]] = {}
    for doc_idx in range(n):
        if doc_idx in uf.parent:
            root = uf.find(doc_idx)
            clusters.setdefault(root, []).append(doc_idx)

    # For each cluster, keep the longest doc (most words), mark rest for removal
    texts = ds["text"]
    to_remove: set[int] = set()
    for members in clusters.values():
        if len(members) < 2:
            continue
        best_idx = max(members, key=lambda i: len(texts[i].split()))
        for idx in members:
            if idx != best_idx:
                to_remove.add(idx)

    return to_remove


@dataclass
class TestDecontaminationIndex:
    content_hashes: set[str]


# ── Helpers ──────────────────────────────────────────────────────────────────


def normalize_text(
    text: str,
    lowercase: bool = True,
) -> str:
    """Normalize text for hashing (NFC, optional lowercase, whitespace collapse)."""
    normalized = unicodedata.normalize("NFC", text or "")
    if lowercase:
        normalized = normalized.lower()
    normalized = _strip_boilerplate_markers(normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _strip_boilerplate_markers(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lower_line = stripped.lower()
        if lower_line in BOILERPLATE_EXACT_MARKERS:
            continue
        if any(lower_line.startswith(prefix) for prefix in BOILERPLATE_PREFIX_MARKERS):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def _strip_html_tags(text: str) -> str:
    has_tags = "<" in text
    has_entities = "&" in text
    if not has_tags and not has_entities:
        return text
    cleaned = text
    if has_tags:
        cleaned = HTML_COMMENT_RE.sub(" ", cleaned)
        cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return html.unescape(cleaned)


def _remove_special_characters(text: str) -> tuple[str, bool]:
    cleaned = text.replace("\t", " ")
    result = _CONTROL_RE.sub("", cleaned)
    return result, len(result) != len(cleaned)


def _looks_like_code(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    code_like = 0
    for line in lines:
        stripped = line.strip()
        # Comment / preprocessor lines
        if stripped.startswith(("#include", "#!", "//", "/*", "*/", "$")):
            code_like += 1
            continue
        # Language keywords co-occurring with code punctuation
        if CODE_KEYWORD_RE.search(stripped) and any(ch in stripped for ch in "{}();[]<>"):
            code_like += 1
            continue
        # JSON/config bracket lines
        if CODE_STRUCTURAL_RE.match(stripped):
            code_like += 1
            continue
        # SQL-like statements
        if stripped.upper().startswith(("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE", "DROP ")):
            code_like += 1
            continue
        # High punctuation density (minified JS, CSS, config)
        punctuation_ratio = sum(ch in "{}();<>/=+-" for ch in stripped) / max(len(stripped), 1)
        if punctuation_ratio > 0.35:
            code_like += 1
            continue
    return code_like / len(lines) > 0.3 or code_like >= 5


# ── Heuristic filter helpers (Phase 3 + 4) ───────────────────────────────────

_CODE_FENCE_SIMPLE_RE = re.compile(r"```")
_HTML_TAG_DENSITY_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WORD_SPLIT_RE = re.compile(r"\b[a-zA-Z_]\w*\b")


def _has_code_fences(text: str) -> bool:
    """Check whether *text* contains triple-backtick code fences."""
    return _CODE_FENCE_SIMPLE_RE.search(text) is not None


def _html_tag_density(text: str) -> float:
    """Fraction of characters occupied by HTML tags."""
    if not text:
        return 0.0
    tag_chars = sum(m.end() - m.start() for m in _HTML_TAG_DENSITY_RE.finditer(text))
    return tag_chars / len(text)


def _symbol_to_text_ratio(text: str) -> float:
    """Fraction of characters in *text* that belong to CODE_SYMBOL_CHARS."""
    if not text:
        return 0.0
    return sum(ch in CODE_SYMBOL_CHARS for ch in text) / len(text)


def _programming_keyword_count(text: str) -> int:
    """Count distinct programming keywords found in *text*."""
    words = {w.lower() for w in _WORD_SPLIT_RE.findall(text)}
    return len(words & PROGRAMMING_KEYWORDS)


def _longest_word_length(text: str) -> int:
    """Length of the longest whitespace-delimited token."""
    words = text.split()
    if not words:
        return 0
    return max(len(w) for w in words)


def _max_ngram_repeats(text: str, n: int = NGRAM_SIZE) -> int:
    """Maximum repetition count of any word-level *n*-gram."""
    words = text.split()
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) if counts else 0


def sanitize_document_text(text: str) -> tuple[str | None, str | None]:
    cleaned = _strip_html_tags(text)
    cleaned, _ = _remove_special_characters(cleaned)
    cleaned = URL_RE.sub("", cleaned)
    cleaned = _strip_boilerplate_markers(cleaned)
    cleaned = MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None, "empty_after_clean"
    if _looks_like_code(cleaned):
        return None, "code_snippet"
    return cleaned, None


def sanitize_batch(texts: list[str]) -> dict[str, list]:
    """Batch wrapper around sanitize_document_text for Dataset.map compatibility.

    Returns dict with 'text' (cleaned or empty string) and 'sanitize_status' columns.
    """
    out_texts: list[str] = []
    statuses: list[str] = []
    for text in texts:
        cleaned, status = sanitize_document_text(text)
        out_texts.append(cleaned or "")
        statuses.append(status or "kept")
    return {"text": out_texts, "sanitize_status": statuses}


# ── Heuristic filter batch functions (Phase 3 + 4) ───────────────────────────


def heuristic_code_filter_batch(texts: list[str]) -> dict[str, list]:
    """Phase 3 heuristic: flag documents that look like code/artifacts.

    Returns {"code_filter_status": ["kept"|"code_fence"|"html_density"|
                                     "symbol_ratio"|"keyword_density"]}
    """
    statuses: list[str] = []
    for text in texts:
        if _has_code_fences(text):
            statuses.append("code_fence")
        elif _html_tag_density(text) > HTML_TAG_DENSITY_THRESHOLD:
            statuses.append("html_density")
        elif _symbol_to_text_ratio(text) > CODE_SYMBOL_RATIO_THRESHOLD:
            statuses.append("symbol_ratio")
        elif _programming_keyword_count(text) > PROGRAMMING_KEYWORD_MAX:
            statuses.append("keyword_density")
        else:
            statuses.append("kept")
    return {"code_filter_status": statuses}


def heuristic_quality_filter_batch(texts: list[str]) -> dict[str, list]:
    """Phase 4 heuristic: flag low-quality / incoherent documents.

    Returns {"quality_filter_status": ["kept"|"too_few_words"|"too_few_chars"|
                                        "unnatural_word"|"ngram_repetition"]}
    """
    statuses: list[str] = []
    for text in texts:
        words = text.split()
        if len(words) < MIN_WORD_COUNT:
            statuses.append("too_few_words")
        elif len(text) < MIN_CHAR_COUNT:
            statuses.append("too_few_chars")
        elif _longest_word_length(text) > MAX_WORD_LENGTH:
            statuses.append("unnatural_word")
        elif _max_ngram_repeats(text, NGRAM_SIZE) > NGRAM_MAX_REPEATS:
            statuses.append("ngram_repetition")
        else:
            statuses.append("kept")
    return {"quality_filter_status": statuses}


# ── Classifier filter batch functions (Phase 3 + 4) ──────────────────────────


def classifier_code_filter_batch(
    texts: list[str], model_path: str
) -> dict[str, list]:
    """Phase 3 classifier: fastText code-vs-prose model.

    Reloads the model each batch (same pattern as detect_language_batch)
    so multiprocessing workers each get their own copy.

    Returns {"code_filter_status": [...], "code_score": [...]}
    """
    model = fasttext.load_model(model_path)
    statuses: list[str] = []
    scores: list[float] = []
    for text in texts:
        first_line = text.split("\n")[0].strip() if text else ""
        if not first_line:
            statuses.append("kept")
            scores.append(0.0)
            continue
        pred = model.predict(first_line)
        label = pred[0][0].replace("__label__", "")
        score = float(pred[1][0])
        if label == "code" and score > CODE_CLASSIFIER_THRESHOLD:
            statuses.append("classifier_code")
            scores.append(score)
        else:
            statuses.append("kept")
            scores.append(score)
    return {"code_filter_status": statuses, "code_score": scores}


def classifier_quality_filter_batch(
    texts: list[str], kenlm_model, edu_model_path: str
) -> dict[str, list]:
    """Phase 4 classifier: KenLM perplexity + fastText educational quality.

    *kenlm_model* must be a pre-loaded kenlm.Model (not fork-safe, so
    num_proc must be 1 when calling this).  *edu_model_path* is loaded
    per-batch like the code classifier.

    Returns {"quality_filter_status": [...], "perplexity": [...], "edu_score": [...]}
    """
    try:
        import kenlm as _kenlm  # noqa: F811
    except ImportError:
        raise ImportError(
            "kenlm is required for classifier quality filtering. "
            "Install with: pip install kenlm"
        )

    edu_model = fasttext.load_model(edu_model_path)
    statuses: list[str] = []
    perplexities: list[float] = []
    edu_scores: list[int] = []

    for text in texts:
        # Perplexity via KenLM
        ppl = kenlm_model.perplexity(text) if text.strip() else float("inf")
        perplexities.append(ppl)

        # Educational quality via fastText
        first_line = text.split("\n")[0].strip() if text else ""
        if first_line:
            pred = edu_model.predict(first_line)
            label = pred[0][0].replace("__label__", "")
            try:
                edu_score = int(label)
            except ValueError:
                edu_score = 0
        else:
            edu_score = 0
        edu_scores.append(edu_score)

        # Filter decisions
        if ppl < KENLM_PERPLEXITY_LOW or ppl > KENLM_PERPLEXITY_HIGH:
            statuses.append("perplexity_outlier")
        elif edu_score < EDUCATIONAL_QUALITY_MIN:
            statuses.append("low_educational")
        else:
            statuses.append("kept")

    return {
        "quality_filter_status": statuses,
        "perplexity": perplexities,
        "edu_score": edu_scores,
    }


def normalize_and_fingerprint(
    text: str, lowercase: bool = True
) -> tuple[str, str | None]:
    """Return normalized text and SHA-1 digest (or ("", None) if empty)."""
    normalized = normalize_text(text, lowercase=lowercase)
    if not normalized:
        return "", None
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return normalized, digest


def normalized_fingerprint(text: str, lowercase: bool = True) -> str | None:
    """Backwards-compatible helper that only returns the fingerprint."""
    _, fingerprint = normalize_and_fingerprint(text, lowercase=lowercase)
    return fingerprint



def detect_language(text: str, model) -> tuple[str, float]:
    """Return (language_code, confidence) using fasttext."""
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return "unknown", 0.0
    pred = model.predict(first_line)
    lang = pred[0][0].replace("__label__", "")
    conf = float(pred[1][0])
    return lang, conf


def detect_language_batch(
    texts: list[str], lang_model_path: str
) -> dict[str, list]:
    """Batch language detection. Loads model from path (safe for multiprocessing).

    Returns dict with 'lang' and 'lang_conf' columns.
    """
    if not texts:
        return {"lang": [], "lang_conf": []}

    model = fasttext.load_model(lang_model_path)
    langs: list[str] = []
    confs: list[float] = []

    # Extract first lines for prediction
    first_lines: list[str] = []
    empty_mask: list[bool] = []
    for text in texts:
        line = text.split("\n")[0].strip() if text else ""
        first_lines.append(line)
        empty_mask.append(not line)

    # fasttext batch predict on non-empty lines
    non_empty_lines = [line for line, is_empty in zip(first_lines, empty_mask) if not is_empty]

    if non_empty_lines:
        pred_labels, pred_scores = model.predict(non_empty_lines)
    else:
        pred_labels, pred_scores = [], []

    pred_idx = 0
    for is_empty in empty_mask:
        if is_empty:
            langs.append("unknown")
            confs.append(0.0)
        else:
            lang = pred_labels[pred_idx][0].replace("__label__", "")
            conf = float(pred_scores[pred_idx][0])
            langs.append(lang)
            confs.append(conf)
            pred_idx += 1

    return {"lang": langs, "lang_conf": confs}


def build_test_decontamination_index(
    data_dir: Path, lowercase: bool = True
) -> TestDecontaminationIndex:
    """Build SHA-1 content hash index from all eval/test splits."""

    content_hashes: set[str] = set()
    split_specs: list[tuple[str, Path]] = []

    wiki_path = data_dir / "wikitext-103-test"
    if wiki_path.exists():
        split_specs.append(("wikitext-103", wiki_path))
    else:
        print(
            f"[decontam] WARNING: Missing Wikitext-103 test split at {wiki_path}."
            " Run the download script."
        )

    owt_eval_path = data_dir / "owt-eval" / "NLP26" / "NLP26_OWT_eval" / "test"
    if owt_eval_path.exists():
        split_specs.append(("nlp26-owt-eval", owt_eval_path))
    else:
        print(
            f"[decontam] WARNING: Missing NLP26 OWT eval split at {owt_eval_path}."
            " Run the download script."
        )

    if not split_specs:
        print("[decontam] No evaluation splits found; contamination checks disabled.")
        return TestDecontaminationIndex(content_hashes=content_hashes)

    for name, path in split_specs:
        print(f"[decontam] Indexing {name} set...")
        dataset = load_from_disk(str(path))
        before_hashes = len(content_hashes)

        _lc = lowercase  # capture for closure

        def _compute_index_batch(batch):
            """Compute fingerprints for a batch."""
            fingerprints: list[str] = []
            for text in batch["text"]:
                if not text or not text.strip():
                    fingerprints.append("")
                    continue
                _, fp = normalize_and_fingerprint(text, lowercase=_lc)
                fingerprints.append(fp or "")
            return {"fingerprint": fingerprints}

        mapped = dataset.map(
            _compute_index_batch,
            batched=True,
            batch_size=256,
            num_proc=1,
        )

        for idx in range(len(mapped)):
            fp = mapped[idx]["fingerprint"]
            if fp:
                content_hashes.add(fp)

        print(f"  -> {len(content_hashes) - before_hashes:,} new hashes")

    print(f"[decontam] Total test hashes: {len(content_hashes):,}")
    return TestDecontaminationIndex(content_hashes=content_hashes)



def decontaminate_batch(
    texts: list[str],
    test_index: TestDecontaminationIndex,
    normalize_lowercase: bool = True,
) -> dict[str, list]:
    """Batch decontamination check via SHA-1 content hash matching.

    Returns dict with 'decontam_status' column.
    Values: 'kept', 'empty_after_normalize', 'test_overlap'.
    """
    if not texts:
        return {"decontam_status": []}

    statuses: list[str] = []
    for text in texts:
        # Normalize + fingerprint
        _, fingerprint = normalize_and_fingerprint(
            text, lowercase=normalize_lowercase
        )
        if fingerprint is None:
            statuses.append("empty_after_normalize")
            continue

        # Exact hash match
        if test_index.content_hashes and fingerprint in test_index.content_hashes:
            statuses.append("test_overlap")
            continue

        statuses.append("kept")

    return {"decontam_status": statuses}


def tokenize_batch(
    texts: list[str], tokenizer
) -> dict[str, list]:
    """Batch tokenization using the fast tokenizer's Rust backend.

    Returns dict with 'input_ids' (list of token ID lists) and 'n_tokens' (lengths).
    """
    if not texts:
        return {"input_ids": [], "n_tokens": []}

    encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
    input_ids = encoded["input_ids"]
    n_tokens = [len(ids) for ids in input_ids]
    return {"input_ids": input_ids, "n_tokens": n_tokens}


# ── Main pipeline ────────────────────────────────────────────────────────────


def preprocess_document(
    text: str,
    tokenizer,
    lang_model,
    test_index: TestDecontaminationIndex,
    lang: str = "en",
    min_tokens: int = 64,
    normalize_lowercase: bool = True,
) -> tuple[list[int] | None, str]:
    """Process one document. Returns (token_ids, status)."""
    if not text or not text.strip():
        return None, "empty"

    text, sanitize_status = sanitize_document_text(text)
    if text is None:
        return None, sanitize_status or "empty_after_clean"

    # Language filter
    detected_lang, conf = detect_language(text, lang_model)
    if detected_lang != lang or conf < 0.8:
        return None, "non_english"

    # Exact-match decontamination using normalized SHA-1 hashes
    _, fingerprint = normalize_and_fingerprint(
        text, lowercase=normalize_lowercase
    )
    if fingerprint is None:
        return None, "empty_after_normalize"
    if test_index.content_hashes and fingerprint in test_index.content_hashes:
        return None, "test_overlap"

    # Tokenize
    tokens = tokenizer.encode(text)

    if len(tokens) < min_tokens:
        return None, "too_short"

    return tokens, "kept"


def run_preprocess(args):
    """Entry point called from main.py --stage preprocess."""
    data_dir = Path(args.data_dir)
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.dataset_size == "small":
        print("[data] Loading 10k subset...")
        ds = load_from_disk(str(data_dir / "openwebtext-10k"))
    elif args.dataset_size == "dummy": 
        print("[data] Loading 100 subset...")
        ds = load_from_disk(str(data_dir / "openwebtext-100"))
    else:
        local_path = data_dir / "openwebtext"
        if local_path.exists():
            print("[data] Loading full OpenWebText from disk...")
            ds = load_from_disk(str(local_path))
        else:
            print("[data] Downloading full OpenWebText (this will take a while)...")
            ds = load_dataset("Skylion007/openwebtext", split="train")

    print(f"[data] Loaded {len(ds):,} documents")

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Language detection model
    lang_model_path = data_dir / "lid.176.ftz"
    if not lang_model_path.exists():
        raise FileNotFoundError(
            f"Language model not found at {lang_model_path}. "
            "Run: uv run python src/scripts/download_data.py"
        )

    # Build decontamination indexes (ngrams + normalized hashes)
    skip_decontam = getattr(args, "skip_decontam", False)
    if skip_decontam:
        print("[decontam] Skipping decontamination (--skip-decontam flag set)")
        test_index = TestDecontaminationIndex(content_hashes=set())
    else:
        test_index = build_test_decontamination_index(
            data_dir, lowercase=FINGERPRINT_LOWERCASE
        )

    # Resolve num_workers
    num_workers = getattr(args, "num_workers", "auto")
    if num_workers == "auto":
        num_workers = os.cpu_count() or 1
    else:
        num_workers = int(num_workers)

    total_docs = len(ds)
    print(f"[preprocess] Processing {total_docs:,} documents (num_workers={num_workers})...")

    # ── Phase 1: Sanitization ────────────────────────────────────────────
    print("\n[phase 1] Sanitization...")
    ds = ds.map(
        lambda batch: sanitize_batch(batch["text"]),
        batched=True,
        batch_size=256,
        num_proc=num_workers,
    )
    before_sanitize = len(ds)
    ds = ds.filter(lambda row: row["sanitize_status"] == "kept", num_proc=num_workers)
    after_sanitize = len(ds)
    print(
        f"  kept {after_sanitize:,} / {before_sanitize:,}"
        f" (removed {before_sanitize - after_sanitize:,})"
    )

    # ── Phase 2: Language filter ─────────────────────────────────────────
    print("\n[phase 2] Language detection...")
    _lang_model_path = str(lang_model_path)
    ds = ds.map(
        lambda batch: detect_language_batch(batch["text"], _lang_model_path),
        batched=True,
        batch_size=256,
        num_proc=num_workers,
    )
    before_lang = len(ds)
    target_lang = args.lang
    ds = ds.filter(
        lambda row: row["lang"] == target_lang and row["lang_conf"] >= 0.8,
        num_proc=num_workers,
    )
    after_lang = len(ds)
    print(
        f"  kept {after_lang:,} / {before_lang:,}"
        f" (removed {before_lang - after_lang:,})"
    )

    # ── Phase 3: Code / artifact removal ─────────────────────────────────
    filter_mode = getattr(args, "filter_mode", "heuristic")
    skip_code_filter = getattr(args, "skip_code_filter", False)
    skip_quality_filter = getattr(args, "skip_quality_filter", False)

    if filter_mode == "none" or skip_code_filter:
        print("\n[phase 3] Code/artifact removal... SKIPPED")
        after_code = len(ds)
    elif filter_mode == "heuristic":
        print("\n[phase 3] Code/artifact removal (heuristic)...")
        ds = ds.map(
            lambda batch: heuristic_code_filter_batch(batch["text"]),
            batched=True,
            batch_size=256,
            num_proc=num_workers,
        )
        before_code = len(ds)
        ds = ds.filter(
            lambda row: row["code_filter_status"] == "kept",
            num_proc=num_workers,
        )
        after_code = len(ds)
        print(
            f"  kept {after_code:,} / {before_code:,}"
            f" (removed {before_code - after_code:,})"
        )
    elif filter_mode == "classifier":
        code_model_path = data_dir / CODE_CLASSIFIER_FILENAME
        if not code_model_path.exists():
            raise FileNotFoundError(
                f"Code classifier model not found at {code_model_path}. "
                "Train it with: uv run python src/scripts/train_filter_models.py code-classifier"
            )
        print("\n[phase 3] Code/artifact removal (classifier)...")
        _code_model_path = str(code_model_path)
        ds = ds.map(
            lambda batch: classifier_code_filter_batch(batch["text"], _code_model_path),
            batched=True,
            batch_size=256,
            num_proc=num_workers,
        )
        before_code = len(ds)
        ds = ds.filter(
            lambda row: row["code_filter_status"] == "kept",
            num_proc=num_workers,
        )
        after_code = len(ds)
        print(
            f"  kept {after_code:,} / {before_code:,}"
            f" (removed {before_code - after_code:,})"
        )

    # ── Phase 4: Quality / coherence filter ───────────────────────────────
    if filter_mode == "none" or skip_quality_filter:
        print("\n[phase 4] Quality/coherence filter... SKIPPED")
        after_quality = len(ds)
    elif filter_mode == "heuristic":
        print("\n[phase 4] Quality/coherence filter (heuristic)...")
        ds = ds.map(
            lambda batch: heuristic_quality_filter_batch(batch["text"]),
            batched=True,
            batch_size=256,
            num_proc=num_workers,
        )
        before_quality = len(ds)
        ds = ds.filter(
            lambda row: row["quality_filter_status"] == "kept",
            num_proc=num_workers,
        )
        after_quality = len(ds)
        print(
            f"  kept {after_quality:,} / {before_quality:,}"
            f" (removed {before_quality - after_quality:,})"
        )
    elif filter_mode == "classifier":
        kenlm_path = data_dir / KENLM_MODEL_FILENAME
        edu_path = data_dir / QUALITY_CLASSIFIER_FILENAME
        if not kenlm_path.exists():
            raise FileNotFoundError(
                f"KenLM model not found at {kenlm_path}. "
                "Train it with: uv run python src/scripts/train_filter_models.py kenlm-model"
            )
        if not edu_path.exists():
            raise FileNotFoundError(
                f"Educational quality model not found at {edu_path}. "
                "Train it with: uv run python src/scripts/train_filter_models.py edu-classifier"
            )
        try:
            import kenlm as _kenlm
        except ImportError:
            raise ImportError(
                "kenlm is required for --filter-mode classifier. "
                "Install with: pip install 'parrotllm[classifier]'"
            )
        print("\n[phase 4] Quality/coherence filter (classifier)...")
        _kenlm_model = _kenlm.Model(str(kenlm_path))
        _edu_path = str(edu_path)
        ds = ds.map(
            lambda batch: classifier_quality_filter_batch(
                batch["text"], _kenlm_model, _edu_path
            ),
            batched=True,
            batch_size=256,
            num_proc=1,  # kenlm not fork-safe
        )
        before_quality = len(ds)
        ds = ds.filter(
            lambda row: row["quality_filter_status"] == "kept",
            num_proc=1,
        )
        after_quality = len(ds)
        print(
            f"  kept {after_quality:,} / {before_quality:,}"
            f" (removed {before_quality - after_quality:,})"
        )

    # ── Phase 5: Fuzzy deduplication ──────────────────────────────────────
    skip_dedup = getattr(args, "skip_dedup", False)
    if skip_dedup:
        print("\n[phase 5] Fuzzy deduplication... SKIPPED")
        after_dedup = len(ds)
    else:
        print("\n[phase 5] Fuzzy deduplication...")
        # Pass 1: compute signatures (parallel)
        ds = ds.map(
            _minhash_signature_batch,
            input_columns=["text"],
            batched=True,
            batch_size=256,
            num_proc=num_workers,
        )
        # Pass 2: build LSH + find duplicates (single-threaded)
        dup_indices = deduplicate_corpus(ds)
        before_dedup = len(ds)
        ds = ds.select([i for i in range(len(ds)) if i not in dup_indices])
        after_dedup = len(ds)
        print(f"  kept {after_dedup:,} / {before_dedup:,} (removed {before_dedup - after_dedup:,} near-duplicates)")

    # ── Phase 6: Decontamination ──────────────────────────────────────────
    if skip_decontam:
        print("\n[phase 6] Decontamination... SKIPPED")
        after_decontam = len(ds)
    else:
        print("\n[phase 6] Decontamination...")
        ds = ds.map(
            lambda batch: decontaminate_batch(batch["text"], test_index),
            batched=True,
            batch_size=256,
            num_proc=num_workers,
        )
        before_decontam = len(ds)
        ds = ds.filter(lambda row: row["decontam_status"] == "kept", num_proc=num_workers)
        after_decontam = len(ds)
        print(
            f"  kept {after_decontam:,} / {before_decontam:,}"
            f" (removed {before_decontam - after_decontam:,})"
        )

    # ── Phase 7: Tokenization ─────────────────────────────────────────────
    print("\n[phase 7] Tokenization...")
    ds = ds.map(
        lambda batch: tokenize_batch(batch["text"], tokenizer),
        batched=True,
        batch_size=256,
        num_proc=1,  # Rust tokenizer parallelizes internally
    )
    before_tok = len(ds)
    ds = ds.filter(lambda row: row["n_tokens"] >= 64, num_proc=1)
    after_tok = len(ds)
    print(
        f"  kept {after_tok:,} / {before_tok:,}"
        f" (removed {before_tok - after_tok:,} short docs)"
    )

    # ── Summary ──────────────────────────────────────────────────────────
    _skip_code = filter_mode == "none" or skip_code_filter
    _skip_quality = filter_mode == "none" or skip_quality_filter
    print(f"\n[preprocess] Filter summary:")
    print(f"  input:            {total_docs:,}")
    print(f"  after sanitize:   {after_sanitize:,}")
    print(f"  after lang:       {after_lang:,}")
    print(f"  after code filter:{after_code:,}{' (skipped)' if _skip_code else ''}")
    print(f"  after quality:    {after_quality:,}{' (skipped)' if _skip_quality else ''}")
    print(f"  after dedup:      {after_dedup:,}{' (skipped)' if skip_dedup else ''}")
    print(f"  after decontam:   {after_decontam:,}{' (skipped)' if skip_decontam else ''}")
    print(f"  after tokenize:   {after_tok:,}")

    # ── Phase 8: Binary output ────────────────────────────────────────────
    print("\n[phase 8] Writing binary output...")
    all_ids = []
    for row in ds:
        all_ids.extend(row["input_ids"])

    token_array = np.array(all_ids, dtype=np.uint16)
    print(f"  Total tokens kept: {len(token_array):,}")

    # Train/val split (99% / 1%)
    n_val = max(1, int(len(token_array) * 0.01))
    val_tokens = token_array[:n_val]
    train_tokens = token_array[n_val:]

    # Save
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_tokens.tofile(str(train_path))
    val_tokens.tofile(str(val_path))

    print(f"\n[done] train: {len(train_tokens):,} tokens ({train_tokens.nbytes / 1e6:.1f} MB) -> {train_path}")
    print(f"[done] val:   {len(val_tokens):,} tokens ({val_tokens.nbytes / 1e6:.1f} MB) -> {val_path}")
