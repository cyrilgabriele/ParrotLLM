"""Annotate OpenWebText documents with educational quality scores (1–5).

This script sends document excerpts to an LLM API and writes results in
fastText format for later training with ``train_filter_models.py edu-classifier``.

Usage:
    export OPENAI_API_KEY=...
    uv run python src/scripts/annotate_educational.py \
        --n-samples 10000 \
        --output data/edu_annotations.txt

Requires the ``openai`` package (not a project dependency — install manually).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path("data")

ANNOTATION_PROMPT = """\
Rate the educational value of the following text on a scale of 1 to 5:
1 = Not educational (ads, spam, nonsense)
2 = Barely educational (news blurbs, social media)
3 = Somewhat educational (blog posts, opinion pieces with some facts)
4 = Educational (well-written articles, textbook excerpts)
5 = Highly educational (academic writing, thorough explanations)

Respond with ONLY a single digit (1-5).

Text:
{text}
"""


def annotate(
    n_samples: int = 10_000,
    output: Path = DATA_DIR / "edu_annotations.txt",
    dataset_path: str | None = None,
) -> None:
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The openai package is required for annotation. "
            "Install with: pip install openai"
        )

    from datasets import load_from_disk, load_dataset

    if dataset_path:
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    client = openai.OpenAI()
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output, "w") as f:
        for sample in ds:
            text = sample["text"]
            if not text or not text.strip():
                continue

            first_line = text.split("\n")[0].strip()
            if not first_line:
                continue

            # Truncate for API efficiency
            excerpt = text[:2000]

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": ANNOTATION_PROMPT.format(text=excerpt)},
                    ],
                    max_tokens=5,
                    temperature=0.0,
                )
                score_text = response.choices[0].message.content.strip()
                score = int(score_text[0])
                if score < 1 or score > 5:
                    continue
            except (ValueError, IndexError, KeyError):
                continue

            f.write(f"__label__{score} {first_line}\n")
            count += 1

            if count % 500 == 0:
                print(f"  annotated {count:,} / {n_samples:,}")

            if count >= n_samples:
                break

    print(f"[annotate] Wrote {count:,} annotations to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate documents with educational quality scores"
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "edu_annotations.txt")
    parser.add_argument("--dataset-path", default=None,
                        help="Path to local HF dataset (default: stream OWT)")
    args = parser.parse_args()
    annotate(n_samples=args.n_samples, output=args.output, dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
