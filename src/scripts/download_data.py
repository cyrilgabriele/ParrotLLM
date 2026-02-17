"""Download all datasets needed for the project."""

import os
import urllib.request
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")


def download_openwebtext_10k():
    """Stream 10k documents from full OpenWebText for fast exploration."""
    out = DATA_DIR / "openwebtext-10k"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    print("[download] Streaming 10k docs from OpenWebText...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    subset = list(ds.take(10_000))
    from datasets import Dataset
    ds_10k = Dataset.from_list(subset)
    ds_10k.save_to_disk(str(out))
    print(f"[done] {len(ds_10k)} documents -> {out}")


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
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    url = "https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K/download"
    print("[download] NLP26 OWT eval split from SWITCHdrive...")
    out.mkdir(exist_ok=True)
    zip_path = out / "owt-eval.zip"
    try:
        urllib.request.urlretrieve(url, str(zip_path))
        # Try to unzip if it's a zip file
        import zipfile
        if zipfile.is_zipfile(str(zip_path)):
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(out))
            zip_path.unlink()
            print(f"[done] Extracted to {out}")
        else:
            print(f"[done] Downloaded to {zip_path} (not a zip, check format)")
    except Exception as e:
        out.rmdir()  # clean up empty dir on failure
        print(f"[warn] NLP26 eval split not available yet: {e}")
        print("  The course may not have uploaded it yet.")
        print("  URL: https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    download_openwebtext_10k()
    download_wikitext103_test()
    download_fasttext_langdetect()
    download_nlp26_eval()

    # Full dataset - uncomment when ready (38GB download)
    # download_openwebtext_full()
