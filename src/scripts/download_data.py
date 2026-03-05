"""Download all datasets needed for the project."""

import argparse
import math
import urllib.request
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")

# ── Subset download constants ─────────────────────────────────────────────────
# Empirical average tokens per raw OpenWebText document (from Section 4 analysis)
_AVG_TOKENS_PER_DOC = 1085
# Estimated fraction of docs removed by the full preprocessing pipeline
_PIPELINE_ATTRITION_RATE = 0.30
NLP26_OWT_EVAL_URLS = (
    "https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K/download",
)


def _download_openwebtext_subset(count: int, suffix: str):
    """Stream a subset of OpenWebText and persist it locally."""
    out = DATA_DIR / f"openwebtext-{suffix}"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    print(f"[download] Streaming {count:,} docs from OpenWebText...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    subset = list(ds.take(count))
    from datasets import Dataset
    ds_subset = Dataset.from_list(subset)
    ds_subset.save_to_disk(str(out))
    print(f"[done] {len(ds_subset)} documents -> {out}")


def download_openwebtext_subset_by_tokens(
    target_tokens: int,
    seed: int = 42,
    attrition_rate: float = _PIPELINE_ATTRITION_RATE,
    avg_tokens_per_doc: float = _AVG_TOKENS_PER_DOC,
    data_dir: Path = DATA_DIR,
) -> Path:
    """Download a seeded random subset of OpenWebText sized to hit a token budget.

    Estimates how many raw documents to fetch as:

        n_docs = ceil(target_tokens / avg_tokens_per_doc / (1 - attrition_rate))

    The shuffle uses a fixed-size buffer (100k) for approximate uniform sampling;
    the same seed always produces the same subset.

    Args:
        target_tokens:    Desired token count in the final processed binary.
        seed:             Random seed for the shuffle.
        attrition_rate:   Fraction of docs expected to be removed by preprocessing.
        avg_tokens_per_doc: Estimated average tokens per raw document.
        data_dir:         Directory to save the downloaded subset.

    Returns:
        Path to the saved dataset directory.
    """
    from datasets import Dataset  # local import keeps module-level imports light

    out = data_dir / f"openwebtext-subset-{target_tokens}-seed{seed}"
    if out.exists():
        print(f"[skip] {out} already exists")
        return out

    n_docs = math.ceil(target_tokens / avg_tokens_per_doc / (1.0 - attrition_rate))
    print(
        f"[download] Streaming ~{n_docs:,} docs from OpenWebText "
        f"(target={target_tokens:,} tokens, seed={seed}, "
        f"attrition={attrition_rate:.0%}, avg_tok/doc={avg_tokens_per_doc})..."
    )

    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=100_000)

    import shutil
    from tqdm import tqdm

    # Stream docs and flush to disk in chunks to avoid materialising the full
    # ~1.3M-document Python list in RAM (~6-7 GB), which silently OOMs on macOS.
    CHUNK_SIZE = 50_000          # ~250 MB per shard in RAM
    chunks_dir = data_dir / f".tmp_chunks_{seed}"
    chunks_dir.mkdir(exist_ok=True)

    chunk_paths: list[Path] = []
    current_chunk: list = []

    with tqdm(total=n_docs, desc="Downloading docs", unit="doc", dynamic_ncols=True) as pbar:
        for doc in ds.take(n_docs):
            current_chunk.append(doc)
            pbar.update(1)
            if len(current_chunk) >= CHUNK_SIZE:
                chunk_path = chunks_dir / f"chunk_{len(chunk_paths):04d}"
                Dataset.from_list(current_chunk).save_to_disk(str(chunk_path))
                chunk_paths.append(chunk_path)
                current_chunk = []

    # Flush any remaining docs
    if current_chunk:
        chunk_path = chunks_dir / f"chunk_{len(chunk_paths):04d}"
        Dataset.from_list(current_chunk).save_to_disk(str(chunk_path))
        chunk_paths.append(chunk_path)

    print(f"  Flushed {len(chunk_paths)} shards — concatenating...")
    from datasets import load_from_disk, concatenate_datasets
    all_chunks = [load_from_disk(str(p)) for p in chunk_paths]
    ds_subset = concatenate_datasets(all_chunks)
    ds_subset.save_to_disk(str(out))
    shutil.rmtree(chunks_dir)
    print(f"[done] {len(ds_subset):,} documents -> {out}")
    return out


def download_openwebtext_10k():
    """Stream 10k documents from full OpenWebText for fast exploration."""
    _download_openwebtext_subset(10_000, "10k")


def download_openwebtext_1k():
    """Stream 1k documents for quick preprocessing smoke tests."""
    _download_openwebtext_subset(1_000, "1k")


def download_openwebtext_100():
    """Stream 100 documents when you only need a few examples."""
    _download_openwebtext_subset(100, "100")


def download_wikitext103_test():
    """Test set for perplexity evaluation."""
    out = DATA_DIR / "wikitext-103-test"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    print("[download] Wikitext-103 test set...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    ds.save_to_disk(str(out))
    print(f"[done] {len(ds)} documents -> {out}")


def download_fasttext_langdetect():
    """FastText language detection model (917KB compressed)."""
    out = DATA_DIR / "lid.176.ftz"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    print("[download] FastText language detection model...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    urllib.request.urlretrieve(url, str(out))
    print(f"[done] -> {out}")


def download_openwebtext_full():
    """Full OpenWebText (~38GB, ~8M documents). Takes a while."""
    out = DATA_DIR / "openwebtext"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    print("[download] Full OpenWebText (this will take a while)...")
    ds = load_dataset("Skylion007/openwebtext", split="train")
    ds.save_to_disk(str(out))
    print(f"[done] {len(ds)} documents -> {out}")


def download_nlp26_eval():
    """NLP26 OWT eval split from SWITCHdrive (for test set decontamination)."""
    out = DATA_DIR / "owt-eval"
    eval_root = out / "NLP26" / "NLP26_OWT_eval"
    test_split = eval_root / "test"
    if test_split.exists():
        print(f"[skip] {test_split} already exists")
        return

    out.mkdir(parents=True, exist_ok=True)
    tmp_archive = out / "nlp26_owt_eval_download.bin"
    last_error: Exception | None = None
    print("[download] NLP26 OWT eval split from SWITCHdrive (full archive)...")

    for url in NLP26_OWT_EVAL_URLS:
        try:
            urllib.request.urlretrieve(url, str(tmp_archive))
            last_error = None
            break
        except Exception as err:  # pragma: no cover - network errors vary
            last_error = err
            print(f"  [warn] Failed to download from {url}: {err}")

    if last_error is not None:
        tmp_archive.unlink(missing_ok=True)
        print("  [warn] NLP26 eval split not available yet.")
        print("  Please verify the SWITCHdrive link provided by the teaching staff.")
        return

    import tarfile
    import zipfile
    try:
        if zipfile.is_zipfile(tmp_archive):
            with zipfile.ZipFile(tmp_archive, "r") as zf:
                zf.extractall(out)
        elif tarfile.is_tarfile(tmp_archive):
            with tarfile.open(tmp_archive, "r:*") as tf:
                tf.extractall(out)
        else:
            # Fallback to shutil for other archive formats (rare).
            import shutil

            shutil.unpack_archive(str(tmp_archive), out)
    except Exception as err:  # pragma: no cover - depends on archive format
        tmp_archive.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to extract NLP26 eval split. Please check the archive format."
        ) from err
    finally:
        tmp_archive.unlink(missing_ok=True)

    if test_split.exists():
        print(f"[done] Extracted eval split to {eval_root}")
    else:
        print(
            "[warn] Extraction finished but expected structure was not found."
            f" Please inspect {out} manually."
        )


DOWNLOAD_TARGETS = {
    "openwebtext-100": download_openwebtext_100,
    "openwebtext-1k": download_openwebtext_1k,
    "openwebtext-10k": download_openwebtext_10k,
    "openwebtext-full": download_openwebtext_full,
    "wikitext103-test": download_wikitext103_test,
    "fasttext-langdetect": download_fasttext_langdetect,
    "nlp26-eval": download_nlp26_eval,
}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        choices=sorted(DOWNLOAD_TARGETS.keys()),
        help="Datasets to download (default: all). Ignored when --target-tokens is set.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Download a seeded random subset sized for this token budget "
            "(e.g. --target-tokens 10000000). "
            "Saves to data/openwebtext-subset-<N>-seed<SEED>."
        ),
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed for the subset shuffle (default: 42).",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(exist_ok=True)

    if args.target_tokens is not None:
        download_openwebtext_subset_by_tokens(
            target_tokens=args.target_tokens,
            seed=args.subset_seed,
        )
        return

    targets = args.targets or list(DOWNLOAD_TARGETS.keys())
    for target in targets:
        DOWNLOAD_TARGETS[target]()


if __name__ == "__main__":
    main()
