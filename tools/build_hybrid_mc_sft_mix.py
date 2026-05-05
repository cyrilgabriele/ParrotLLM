"""Build the v9 hybrid MC SFT mix.

The goal is not to replace Alpaca SFT with synthetic data. The main signal
comes from public TRAIN splits that match the benchmark task families. A small
programmatic control layer is added only to make the raw MC format and answer
letter balance explicit.

Output rows use the existing RawCompletionTemplate contract:

    {"instruction": "Context: ...\nA) ...\nB) ...\nAnswer:", "response": " B"}

Training-time decontamination still runs in build_sft_datasets. This builder
also filters against the same benchmark hash registry as a belt-and-suspenders
check while creating the JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = Path("data/synthetic/sft_v9_mc_balanced.jsonl")
SEED = 47
FOUR_WAY_LETTERS = "ABCD"


@dataclass(frozen=True)
class MCItem:
    source: str
    stem: str
    choices: tuple[str, ...]
    answer_idx: int


def _normalise_for_hash(text: str) -> str:
    return " ".join((text or "").lower().split())


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_decontam_hashes() -> set[str]:
    from src.post_training.sft.data import DECONTAM_LOADERS, build_decontam_index

    texts: list[str] = []
    for name, loader in DECONTAM_LOADERS.items():
        try:
            texts.extend(loader())
            print(f"  decontam: loaded {name}")
        except Exception as exc:
            print(f"  decontam: skipped {name}: {exc!r}")
    return build_decontam_index(texts)


def _render_raw_mc(
    item: MCItem,
    *,
    rng: random.Random,
    force_answer_letter: str | None = None,
) -> dict | None:
    choices = [c.strip() for c in item.choices if c and c.strip()]
    if not item.stem.strip() or item.answer_idx < 0 or item.answer_idx >= len(choices):
        return None
    if len(choices) < 2:
        return None

    answer = choices[item.answer_idx]
    distractors = [c for i, c in enumerate(choices) if i != item.answer_idx]
    rng.shuffle(distractors)

    if len(choices) >= 4:
        letters = FOUR_WAY_LETTERS
        target_letter = (
            force_answer_letter
            if force_answer_letter is not None and force_answer_letter in letters
            else rng.choice(letters)
        )
        target_idx = letters.index(target_letter)
        picked = distractors[:3]
        if len(picked) < 3:
            return None
        rendered_choices = picked[:]
        rendered_choices.insert(target_idx, answer)
    else:
        letters = "AB"
        target_letter = (
            force_answer_letter
            if force_answer_letter is not None and force_answer_letter in letters
            else rng.choice(letters)
        )
        target_idx = letters.index(target_letter)
        if not distractors:
            return None
        rendered_choices = [distractors[0]]
        rendered_choices.insert(target_idx, answer)

    option_block = "\n".join(
        f"{letters[i]}) {rendered_choices[i]}" for i in range(len(rendered_choices))
    )
    instruction = f"Context: {item.stem.strip()}\n{option_block}\nAnswer:"
    return {
        "instruction": instruction,
        "response": f" {letters[target_idx]}",
        "source": item.source,
    }


def _is_clean(row: dict, decontam: set[str]) -> bool:
    blob = f"{row['instruction']} {row['response']}"
    return _sha1(_normalise_for_hash(blob)) not in decontam


def _sample_rendered(
    items: list[MCItem],
    *,
    quota: int,
    rng: random.Random,
    decontam: set[str],
    balance_four_way: bool = True,
) -> list[dict]:
    rng.shuffle(items)
    out: list[dict] = []
    target_cycle = list(FOUR_WAY_LETTERS)
    cycle_idx = 0

    for item in items:
        if len(out) >= quota:
            break
        force = None
        if balance_four_way and len(item.choices) >= 4:
            force = target_cycle[cycle_idx % len(target_cycle)]
            cycle_idx += 1
        row = _render_raw_mc(item, rng=rng, force_answer_letter=force)
        if row is None or not _is_clean(row, decontam):
            continue
        out.append(row)
    return out


def load_hellaswag() -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="train")
    items: list[MCItem] = []
    for row in ds:
        try:
            label = int(row.get("label", -1))
        except Exception:
            continue
        stem = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
        items.append(MCItem("hellaswag_train", stem, tuple(row.get("endings", [])), label))
    return items


def load_openbookqa() -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split="train")
    items: list[MCItem] = []
    for row in ds:
        labels = list(row["choices"]["label"])
        choices = tuple(row["choices"]["text"])
        answer = row["answerKey"]
        if answer in labels:
            items.append(MCItem("openbookqa_train", row["question_stem"], choices, labels.index(answer)))
    return items


def load_winogrande() -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/winogrande",
        "winogrande_xl",
        split="train",
        trust_remote_code=True,
    )
    items: list[MCItem] = []
    for row in ds:
        answer = str(row.get("answer", "")).strip()
        if answer in {"1", "2"}:
            items.append(
                MCItem(
                    "winogrande_train",
                    row["sentence"],
                    (row["option1"], row["option2"]),
                    int(answer) - 1,
                )
            )
    return items


def load_sciq() -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="train")
    return [
        MCItem(
            "sciq_train",
            row["question"],
            (
                row["correct_answer"],
                row["distractor1"],
                row["distractor2"],
                row["distractor3"],
            ),
            0,
        )
        for row in ds
    ]


def load_arc(config_name: str, source: str) -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", config_name, split="train")
    items: list[MCItem] = []
    for row in ds:
        labels = list(row["choices"]["label"])
        choices = tuple(row["choices"]["text"])
        answer = row["answerKey"]
        if answer in labels:
            items.append(MCItem(source, row["question"], choices, labels.index(answer)))
    return items


def load_commonsenseqa() -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="train")
    items: list[MCItem] = []
    for row in ds:
        labels = list(row["choices"]["label"])
        choices = tuple(row["choices"]["text"])
        answer = row["answerKey"]
        if answer in labels:
            items.append(MCItem("commonsenseqa_train", row["question"], choices, labels.index(answer)))
    return items


def load_boolq_control() -> list[MCItem]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="train")
    items: list[MCItem] = []
    for row in ds:
        answer = bool(row["answer"])
        stem = f"{row['passage']}\nQuestion: {row['question']}"
        items.append(MCItem("boolq_train_control", stem, ("yes", "no"), 0 if answer else 1))
    return items


def programmatic_control_items() -> list[MCItem]:
    facts = [
        ("What is the capital of France?", ("Paris", "Madrid", "Berlin", "Rome"), 0),
        ("Which planet is known as the Red Planet?", ("Mars", "Venus", "Jupiter", "Mercury"), 0),
        ("What gas do plants absorb from the air?", ("carbon dioxide", "oxygen", "helium", "nitrogen"), 0),
        ("How many sides does a triangle have?", ("three", "four", "five", "six"), 0),
        ("What is H2O commonly called?", ("water", "salt", "oxygen", "sugar"), 0),
        ("Which object is used to tell time?", ("clock", "spoon", "shoe", "blanket"), 0),
        ("What color is grass usually?", ("green", "blue", "red", "black"), 0),
        ("Which number is even?", ("eight", "seven", "nine", "eleven"), 0),
    ]
    return [
        MCItem("programmatic_control", stem, choices, answer_idx)
        for stem, choices, answer_idx in facts
    ]


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            payload = {
                "instruction": row["instruction"],
                "response": row["response"],
                "source": row["source"],
            }
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target", type=int, default=7200)
    parser.add_argument("--control", type=int, default=600)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("Loading decontamination hashes...")
    decontam = _load_decontam_hashes()
    print(f"  decontam: {len(decontam)} hashes")

    quotas = {
        "hellaswag": 1500,
        "openbookqa": 1600,
        "winogrande": 1000,
        "sciq": 700,
        "arc_easy": 500,
        "arc_challenge": 300,
        "commonsenseqa": 1200,
        "boolq_control": 400,
    }
    scale = max(0.1, (args.target - args.control) / sum(quotas.values()))
    quotas = {name: max(1, int(round(value * scale))) for name, value in quotas.items()}

    source_loaders = {
        "hellaswag": load_hellaswag,
        "openbookqa": load_openbookqa,
        "winogrande": load_winogrande,
        "sciq": load_sciq,
        "arc_easy": lambda: load_arc("ARC-Easy", "arc_easy_train"),
        "arc_challenge": lambda: load_arc("ARC-Challenge", "arc_challenge_train"),
        "commonsenseqa": load_commonsenseqa,
        "boolq_control": load_boolq_control,
    }

    rows: list[dict] = []
    for name, quota in quotas.items():
        print(f"Loading {name} train split...")
        try:
            items = source_loaders[name]()
        except Exception as exc:
            print(f"  skipped {name}: {exc!r}")
            continue
        picked = _sample_rendered(
            items,
            quota=quota,
            rng=rng,
            decontam=decontam,
            balance_four_way=True,
        )
        rows.extend(picked)
        print(f"  added {len(picked)} rows")

    control_rows: list[dict] = []
    control_items = programmatic_control_items()
    while len(control_rows) < args.control:
        picked = _sample_rendered(
            control_items[:],
            quota=len(control_items),
            rng=rng,
            decontam=decontam,
            balance_four_way=True,
        )
        control_rows.extend(picked)
    rows.extend(control_rows[: args.control])

    rng.shuffle(rows)
    _write_jsonl(rows, args.out)

    letters = Counter(row["response"].strip() for row in rows)
    sources = Counter(row["source"] for row in rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Answer distribution: {dict(sorted(letters.items()))}")
    print(f"Source distribution: {dict(sorted(sources.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
