"""Download all datasets needed for the project."""

import argparse
import urllib.request
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")
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
        help="Datasets to download (default: all).",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(exist_ok=True)

    targets = args.targets or list(DOWNLOAD_TARGETS.keys())
    for target in targets:
        DOWNLOAD_TARGETS[target]()


if __name__ == "__main__":
    main()
