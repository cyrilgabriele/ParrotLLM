"""Train classifier models for the content-filtering pipeline.

Usage:
    uv run python src/scripts/train_filter_models.py code-classifier
    uv run python src/scripts/train_filter_models.py kenlm-model
    uv run python src/scripts/train_filter_models.py edu-classifier

Models are saved into ``data/``.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

DATA_DIR = Path("data")


# ── code-classifier ──────────────────────────────────────────────────────────


def train_code_classifier(
    n_samples: int = 50_000,
    output: Path = DATA_DIR / "code_vs_prose.ftz",
) -> None:
    """Train a fastText binary classifier (code vs prose).

    Positive class (__label__code): samples from The Stack.
    Negative class (__label__prose): samples from Wikipedia.
    """
    from datasets import load_dataset

    print(f"[code-classifier] Streaming {n_samples} code samples from The Stack...")
    code_ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train",
        streaming=True,
    )

    print(f"[code-classifier] Streaming {n_samples} prose samples from Wikipedia...")
    prose_ds = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name
        count = 0
        for sample in code_ds:
            first_line = sample["content"].split("\n")[0].strip()
            if first_line:
                f.write(f"__label__code {first_line}\n")
                count += 1
            if count >= n_samples:
                break

        count = 0
        for sample in prose_ds:
            first_line = sample["text"].split("\n")[0].strip()
            if first_line:
                f.write(f"__label__prose {first_line}\n")
                count += 1
            if count >= n_samples:
                break

    import fasttext

    print("[code-classifier] Training fastText model...")
    model = fasttext.train_supervised(
        input=tmp_path,
        epoch=5,
        lr=0.5,
        wordNgrams=2,
        dim=100,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))
    print(f"[code-classifier] Saved to {output}")
    Path(tmp_path).unlink()


# ── kenlm-model ──────────────────────────────────────────────────────────────


def train_kenlm_model(
    n_articles: int = 1_000_000,
    output: Path = DATA_DIR / "wikipedia_en.arpa.bin",
) -> None:
    """Train a KenLM 5-gram model on Wikipedia text.

    Requires ``kenlm`` CLI tools (``lmplz``, ``build_binary``) on PATH.
    """
    from datasets import load_dataset

    print(f"[kenlm-model] Streaming {n_articles:,} Wikipedia articles...")
    ds = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
    )

    arpa_path = output.with_suffix(".arpa")
    text_path = output.with_suffix(".txt")

    with open(text_path, "w") as f:
        count = 0
        for sample in ds:
            text = sample["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
                count += 1
            if count >= n_articles:
                break

    print("[kenlm-model] Training 5-gram ARPA model with lmplz...")
    subprocess.run(
        ["lmplz", "-o", "5", "--text", str(text_path), "--arpa", str(arpa_path)],
        check=True,
    )

    print("[kenlm-model] Converting to binary with build_binary...")
    subprocess.run(
        ["build_binary", str(arpa_path), str(output)],
        check=True,
    )

    text_path.unlink()
    arpa_path.unlink()
    print(f"[kenlm-model] Saved to {output}")


# ── edu-classifier ───────────────────────────────────────────────────────────


def train_edu_classifier(
    annotations: Path = DATA_DIR / "edu_annotations.txt",
    output: Path = DATA_DIR / "educational_quality.ftz",
) -> None:
    """Train a fastText educational-quality classifier (labels 1–5).

    Expects *annotations* in fastText format::

        __label__3 First line of document...
        __label__5 Another first line...

    Generate these with ``annotate_educational.py``.
    """
    if not annotations.exists():
        raise FileNotFoundError(
            f"Annotation file not found at {annotations}. "
            "Generate it first: uv run python src/scripts/annotate_educational.py"
        )

    import fasttext

    print("[edu-classifier] Training fastText model...")
    model = fasttext.train_supervised(
        input=str(annotations),
        epoch=10,
        lr=0.3,
        wordNgrams=2,
        dim=100,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))
    print(f"[edu-classifier] Saved to {output}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifier-mode filter models")
    parser.add_argument(
        "target",
        choices=["code-classifier", "kenlm-model", "edu-classifier"],
        help="Which model to train",
    )
    args = parser.parse_args()

    if args.target == "code-classifier":
        train_code_classifier()
    elif args.target == "kenlm-model":
        train_kenlm_model()
    elif args.target == "edu-classifier":
        train_edu_classifier()


if __name__ == "__main__":
    main()
