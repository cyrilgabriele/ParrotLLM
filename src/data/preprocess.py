"""Data preprocessing pipeline: filter, tokenize, and save as binary splits."""

import os
from collections import Counter
from pathlib import Path

import fasttext
import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import GPT2TokenizerFast


# ── Helpers ──────────────────────────────────────────────────────────────────


def detect_language(text: str, model) -> tuple[str, float]:
    """Return (language_code, confidence) using fasttext."""
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return "unknown", 0.0
    pred = model.predict(first_line)
    lang = pred[0][0].replace("__label__", "")
    conf = float(pred[1][0])
    return lang, conf


def extract_ngrams(text: str, n: int = 13) -> set[str]:
    """Extract character-level n-grams for contamination detection."""
    text = text.lower().strip()
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_test_ngrams(data_dir: Path) -> set[str]:
    """Build n-gram index from all test/eval sets for decontamination."""
    ngrams: set[str] = set()

    # Wikitext-103 test
    wiki_path = data_dir / "wikitext-103-test"
    if wiki_path.exists():
        print("[decontam] Indexing Wikitext-103 test set...")
        wiki_test = load_from_disk(str(wiki_path))
        for doc in tqdm(wiki_test, desc="  wikitext-103"):
            if doc["text"].strip():
                ngrams.update(extract_ngrams(doc["text"]))
        print(f"  -> {len(ngrams):,} n-grams")

    # NLP26 OWT eval split
    owt_eval_path = data_dir / "owt-eval" / "NLP26" / "NLP26_OWT_eval" / "test"
    if owt_eval_path.exists():
        print("[decontam] Indexing NLP26 OWT eval set...")
        owt_eval = load_from_disk(str(owt_eval_path))
        before = len(ngrams)
        for doc in tqdm(owt_eval, desc="  owt-eval"):
            if doc["text"].strip():
                ngrams.update(extract_ngrams(doc["text"]))
        print(f"  -> {len(ngrams) - before:,} new n-grams")

    print(f"[decontam] Total test n-grams: {len(ngrams):,}")
    return ngrams


def is_contaminated(
    text: str, test_ngrams: set[str], n: int = 13, threshold: float = 0.8
) -> bool:
    """Check if a document has high overlap with test set n-grams."""
    doc_ngrams = extract_ngrams(text, n)
    if not doc_ngrams:
        return False
    overlap = len(doc_ngrams & test_ngrams) / len(doc_ngrams)
    return overlap > threshold


# ── Main pipeline ────────────────────────────────────────────────────────────


def preprocess_document(
    text: str,
    tokenizer,
    lang_model,
    test_ngrams: set[str],
    lang: str = "en",
    min_tokens: int = 64,
) -> tuple[list[int] | None, str]:
    """Process one document. Returns (token_ids, status)."""
    if not text.strip():
        return None, "empty"

    # Language filter
    detected_lang, conf = detect_language(text, lang_model)
    if detected_lang != lang or conf < 0.5:
        return None, "non_english"

    # Decontamination
    if test_ngrams and is_contaminated(text, test_ngrams):
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

    # Build decontamination index
    test_ngrams = build_test_ngrams(data_dir)

    # Process all documents
    print(f"[preprocess] Processing {len(ds):,} documents...")
    stats: Counter = Counter()
    all_tokens: list[int] = []

    for doc in tqdm(ds, desc="preprocessing"):
        tokens, status = preprocess_document(
            doc["text"], tokenizer, lang_model, test_ngrams, lang=args.lang
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
