from __future__ import annotations

"""Data preprocessing pipeline: filter, tokenize, and save as binary splits."""

import hashlib
import html
import random
import re
import unicodedata
from collections import Counter, defaultdict
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
HTML_TAG_RE = re.compile(r"<[^>]+>")
CODE_KEYWORD_RE = re.compile(
    r"\b(class|def|function|public|private|import|from|package|namespace|var|let|const|return|for|while|if|else|try|catch|using|struct|enum)\b",
    re.IGNORECASE,
)
CODE_FENCE_RE = re.compile(r"```|~~~|<code>|</code>|&lt;/?code&gt;", re.IGNORECASE)

ENABLE_MINHASH_DECONTAMINATION = True
MINHASH_NUM_PERMUTATIONS = 64
MINHASH_ROWS_PER_BAND = 4  # results in 16 bands
MINHASH_SHINGLE_SIZE = 5
MINHASH_JACCARD_THRESHOLD = 0.8
MINHASH_PRIME = 2_305_843_009_213_693_951  # 2^61 - 1 (Mersenne prime)
MINHASH_MAX_HASH = MINHASH_PRIME - 1
MINHASH_SEED = 1337


class MinHasher:
    """Lightweight MinHash implementation with pre-generated permutations."""

    def __init__(self, num_perm: int, seed: int = 0) -> None:
        self.num_perm = num_perm
        self.prime = MINHASH_PRIME
        rng = random.Random(seed)
        self.permutations = [
            (rng.randrange(1, self.prime - 1), rng.randrange(0, self.prime - 1))
            for _ in range(num_perm)
        ]

    def signature(self, hashes: set[int]) -> tuple[int, ...]:
        if not hashes:
            return tuple([MINHASH_MAX_HASH] * self.num_perm)
        sig: list[int] = []
        for a, b in self.permutations:
            min_val = MINHASH_MAX_HASH
            for value in hashes:
                hashed = (a * value + b) % self.prime
                if hashed < min_val:
                    min_val = hashed
            sig.append(min_val)
        return tuple(sig)


class MinHashLSHIndex:
    """Banding-based LSH for MinHash signatures."""

    def __init__(self, num_perm: int, rows_per_band: int) -> None:
        if num_perm % rows_per_band != 0:
            raise ValueError("num_perm must be divisible by rows_per_band")
        self.num_perm = num_perm
        self.rows_per_band = rows_per_band
        self.num_bands = num_perm // rows_per_band
        self.bands: list[dict[tuple[int, ...], set[str]]] = [
            defaultdict(set) for _ in range(self.num_bands)
        ]

    def insert(self, doc_id: str, signature: tuple[int, ...]) -> None:
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_key = tuple(signature[start : start + self.rows_per_band])
            self.bands[band_idx][band_key].add(doc_id)

    def query(self, signature: tuple[int, ...]) -> set[str]:
        candidates: set[str] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_key = tuple(signature[start : start + self.rows_per_band])
            hits = self.bands[band_idx].get(band_key)
            if hits:
                candidates.update(hits)
        return candidates


MINHASHER = (
    MinHasher(MINHASH_NUM_PERMUTATIONS, seed=MINHASH_SEED)
    if ENABLE_MINHASH_DECONTAMINATION
    else None
)


@dataclass
class TestDecontaminationIndex:
    ngrams: set[str]
    content_hashes: set[str]
    minhash_index: MinHashLSHIndex | None = None
    doc_shingles: dict[str, set[int]] | None = None


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
    if "<" not in text and "&lt;" not in text.lower():
        return text
    no_tags = HTML_TAG_RE.sub(" ", text)
    return html.unescape(no_tags)


def _remove_special_characters(text: str) -> tuple[str, bool]:
    cleaned_chars: list[str] = []
    removed = False
    for ch in text:
        if ch in {"\ufffd", "\ufeff"}:
            removed = True
            continue
        if ch == "\t":
            cleaned_chars.append(" ")
            continue
        if ch == "\n":
            cleaned_chars.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            removed = True
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars), removed


def _looks_like_code(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    code_like = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#include", "//", "/*", "*/")):
            code_like += 1
            continue
        if CODE_KEYWORD_RE.search(stripped) and any(ch in stripped for ch in "{}();[]<>"):
            code_like += 1
            continue
        punctuation_ratio = sum(ch in "{}();<>/=+-" for ch in stripped) / max(len(stripped), 1)
        if punctuation_ratio > 0.35:
            code_like += 1
            continue
    return code_like / len(lines) > 0.3 or code_like >= 5


def sanitize_document_text(text: str) -> tuple[str | None, str | None]:
    cleaned = _strip_html_tags(text)
    cleaned, _ = _remove_special_characters(cleaned)
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


def compute_shingle_hashes(normalized_text: str, shingle_size: int) -> set[int]:
    tokens = normalized_text.split()
    if len(tokens) < shingle_size:
        return set()
    shingles: set[int] = set()
    for idx in range(len(tokens) - shingle_size + 1):
        shingle = " ".join(tokens[idx : idx + shingle_size])
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        shingles.add(int.from_bytes(digest, "big") % MINHASH_PRIME)
    return shingles


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    union = len(a | b)
    return intersection / union


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


def extract_ngrams(text: str, n: int = 13) -> set[str]:
    """Extract character-level n-grams for contamination detection."""
    text = text.lower().strip()
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_test_decontamination_index(
    data_dir: Path, lowercase: bool = True
) -> TestDecontaminationIndex:
    """Build n-gram and normalized SHA-1 indexes for all eval/test splits."""

    ngrams: set[str] = set()
    content_hashes: set[str] = set()
    split_specs: list[tuple[str, Path]] = []
    minhash_index = (
        MinHashLSHIndex(MINHASH_NUM_PERMUTATIONS, MINHASH_ROWS_PER_BAND)
        if ENABLE_MINHASH_DECONTAMINATION and MINHASHER is not None
        else None
    )
    doc_shingles: dict[str, set[int]] | None = {} if minhash_index else None

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
        return TestDecontaminationIndex(
            ngrams=ngrams,
            content_hashes=content_hashes,
            minhash_index=minhash_index,
            doc_shingles=doc_shingles,
        )

    for name, path in split_specs:
        print(f"[decontam] Indexing {name} set...")
        dataset = load_from_disk(str(path))
        before = len(ngrams)
        before_hashes = len(content_hashes)

        _lc = lowercase  # capture for closure

        def _compute_index_batch(batch):
            """Compute ngrams, fingerprints, and normalized text for a batch."""
            all_ngrams: list[str] = []
            fingerprints: list[str] = []
            normalized_texts: list[str] = []
            for text in batch["text"]:
                if not text or not text.strip():
                    all_ngrams.append("")
                    fingerprints.append("")
                    normalized_texts.append("")
                    continue
                norm_text, fp = normalize_and_fingerprint(text, lowercase=_lc)
                if not norm_text:
                    all_ngrams.append("")
                    fingerprints.append("")
                    normalized_texts.append("")
                    continue
                doc_ngrams = extract_ngrams(text)
                # Join ngrams with newline for serializability
                all_ngrams.append("\n".join(doc_ngrams) if doc_ngrams else "")
                fingerprints.append(fp or "")
                normalized_texts.append(norm_text)
            return {
                "ngrams_joined": all_ngrams,
                "fingerprint": fingerprints,
                "normalized_text": normalized_texts,
            }

        mapped = dataset.map(
            _compute_index_batch,
            batched=True,
            batch_size=256,
            num_proc=1,
        )

        # Reduce: union ngrams, collect hashes, insert MinHash signatures
        for idx in range(len(mapped)):
            row = mapped[idx]
            ngrams_str = row["ngrams_joined"]
            if ngrams_str:
                ngrams.update(ngrams_str.split("\n"))
            fp = row["fingerprint"]
            if fp:
                content_hashes.add(fp)
            norm_text = row["normalized_text"]
            if (
                minhash_index
                and doc_shingles is not None
                and MINHASHER is not None
                and norm_text
            ):
                shingle_hashes = compute_shingle_hashes(
                    norm_text, MINHASH_SHINGLE_SIZE
                )
                if not shingle_hashes:
                    continue
                signature = MINHASHER.signature(shingle_hashes)
                doc_id = f"{name}:{idx}"
                minhash_index.insert(doc_id, signature)
                doc_shingles[doc_id] = shingle_hashes

        print(
            f"  -> {len(ngrams) - before:,} new n-grams,"
            f" {len(content_hashes) - before_hashes:,} new hashes"
        )

    print(f"[decontam] Total test n-grams: {len(ngrams):,}")
    print(f"[decontam] Total test hashes: {len(content_hashes):,}")
    if minhash_index and doc_shingles is not None:
        print(
            f"[decontam] MinHash protected docs: {len(doc_shingles):,}"
            f" (shingle size={MINHASH_SHINGLE_SIZE}, perms={MINHASH_NUM_PERMUTATIONS})"
        )
    return TestDecontaminationIndex(
        ngrams=ngrams,
        content_hashes=content_hashes,
        minhash_index=minhash_index,
        doc_shingles=doc_shingles,
    )


def build_test_ngrams(data_dir: Path) -> set[str]:
    """Backwards-compatible wrapper returning only the n-gram index."""
    return build_test_decontamination_index(data_dir).ngrams


def is_contaminated(
    text: str, test_ngrams: set[str], n: int = 13, threshold: float = 0.8
) -> bool:
    """Check if a document has high overlap with test set n-grams."""
    doc_ngrams = extract_ngrams(text, n)
    if not doc_ngrams:
        return False
    overlap = len(doc_ngrams & test_ngrams) / len(doc_ngrams)
    return overlap > threshold


def decontaminate_batch(
    texts: list[str],
    test_index: TestDecontaminationIndex,
    normalize_lowercase: bool = True,
) -> dict[str, list]:
    """Batch decontamination check: hash match, MinHash overlap, n-gram contamination.

    Returns dict with 'decontam_status' column.
    Values: 'kept', 'empty_after_normalize', 'test_overlap', 'minhash_overlap', 'contaminated'.
    """
    if not texts:
        return {"decontam_status": []}

    statuses: list[str] = []
    for text in texts:
        # Normalize + fingerprint
        normalized_text, fingerprint = normalize_and_fingerprint(
            text, lowercase=normalize_lowercase
        )
        if fingerprint is None:
            statuses.append("empty_after_normalize")
            continue

        # Exact hash match
        if test_index.content_hashes and fingerprint in test_index.content_hashes:
            statuses.append("test_overlap")
            continue

        # MinHash/LSH overlap
        if (
            ENABLE_MINHASH_DECONTAMINATION
            and test_index.minhash_index is not None
            and test_index.doc_shingles is not None
            and MINHASHER is not None
        ):
            shingle_hashes = compute_shingle_hashes(
                normalized_text, MINHASH_SHINGLE_SIZE
            )
            if shingle_hashes:
                signature = MINHASHER.signature(shingle_hashes)
                candidates = test_index.minhash_index.query(signature)
                found_overlap = False
                if candidates:
                    for candidate in candidates:
                        target = test_index.doc_shingles.get(candidate)
                        if not target:
                            continue
                        similarity = jaccard_similarity(shingle_hashes, target)
                        if similarity >= MINHASH_JACCARD_THRESHOLD:
                            found_overlap = True
                            break
                if found_overlap:
                    statuses.append("minhash_overlap")
                    continue

        # N-gram contamination
        if test_index.ngrams and is_contaminated(text, test_index.ngrams):
            statuses.append("contaminated")
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
    if detected_lang != lang or conf < 0.5:
        return None, "non_english"

    # Exact-match decontamination using normalized hashes
    normalized_text, fingerprint = normalize_and_fingerprint(
        text, lowercase=normalize_lowercase
    )
    if fingerprint is None:
        return None, "empty_after_normalize"
    if test_index.content_hashes and fingerprint in test_index.content_hashes:
        return None, "test_overlap"

    # MinHash/LSH overlap detection
    if (
        ENABLE_MINHASH_DECONTAMINATION
        and test_index.minhash_index is not None
        and test_index.doc_shingles is not None
        and MINHASHER is not None
    ):
        shingle_hashes = compute_shingle_hashes(normalized_text, MINHASH_SHINGLE_SIZE)
        if shingle_hashes:
            signature = MINHASHER.signature(shingle_hashes)
            candidates = test_index.minhash_index.query(signature)
            if candidates:
                for candidate in candidates:
                    target = test_index.doc_shingles.get(candidate)
                    if not target:
                        continue
                    similarity = jaccard_similarity(shingle_hashes, target)
                    if similarity >= MINHASH_JACCARD_THRESHOLD:
                        return None, "minhash_overlap"

    # Decontamination
    if test_index.ngrams and is_contaminated(text, test_index.ngrams):
        return None, "contaminated"

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
    lang_model = fasttext.load_model(str(lang_model_path))

    # Build decontamination indexes (ngrams + normalized hashes)
    test_index = build_test_decontamination_index(
        data_dir, lowercase=FINGERPRINT_LOWERCASE
    )

    # Process all documents
    print(f"[preprocess] Processing {len(ds):,} documents...")
    stats: Counter = Counter()
    all_tokens: list[int] = []

    for doc in tqdm(ds, desc="preprocessing"):
        tokens, status = preprocess_document(
            doc["text"],
            tokenizer,
            lang_model,
            test_index,
            lang=args.lang,
            normalize_lowercase=FINGERPRINT_LOWERCASE,
        )
        stats[status] += 1
        if tokens is not None:
            all_tokens.extend(tokens)

    # Stats
    print("\n[preprocess] Filter stats:")
    for status, count in stats.most_common():
        print(f"  {status}: {count:,} ({count / len(ds) * 100:.1f}%)")
    print(f"  Total tokens kept: {len(all_tokens):,}")

    # Convert to numpy uint16 (vocab 50257 fits in uint16 max 65535)
    token_array = np.array(all_tokens, dtype=np.uint16)

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
