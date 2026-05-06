"""Build raw-format MC preference pairs for DPO v9.

Rows use the DPO local preference JSONL contract:

    {"prompt": "...\\nAnswer:", "chosen": " B", "rejected": " C", "template": "raw"}

The prompt is byte-compatible with the leaderboard runner. The chosen
completion is the correct answer letter; rejected is one shuffled wrong
letter from the same rendered option set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.post_training.hf_cache import cleanup_hf_dataset_cache
from tools.build_hybrid_mc_sft_mix import (
    SEED,
    _is_clean,
    _load_decontam_hashes,
    _render_raw_mc,
    load_arc,
    load_boolq_control,
    load_commonsenseqa,
    load_hellaswag,
    load_openbookqa,
    load_sciq,
    load_winogrande,
)


OUT_PATH = Path("data/synthetic/dpo_v9_mc_preferences.jsonl")
_OPTION_RE = re.compile(r"^([A-D])\)", re.MULTILINE)


def _to_preference(row: dict, *, rng: random.Random) -> dict | None:
    letters = _OPTION_RE.findall(row["instruction"])
    correct = row["response"].strip()
    wrong = [letter for letter in letters if letter != correct]
    if correct not in letters or not wrong:
        return None
    return {
        "prompt": row["instruction"],
        "chosen": f" {correct}",
        "rejected": f" {rng.choice(wrong)}",
        "template": "raw",
        "source": row["source"],
    }


def _sample_preferences(
    items,
    *,
    quota: int,
    rng: random.Random,
    decontam: set[str],
) -> list[dict]:
    rng.shuffle(items)
    rows: list[dict] = []
    force_cycle = "ABCD"
    cycle_idx = 0
    for item in items:
        if len(rows) >= quota:
            break
        force = force_cycle[cycle_idx % len(force_cycle)] if len(item.choices) >= 4 else None
        cycle_idx += 1
        rendered = _render_raw_mc(item, rng=rng, force_answer_letter=force)
        if rendered is None or not _is_clean(rendered, decontam):
            continue
        pref = _to_preference(rendered, rng=rng)
        if pref is not None:
            rows.append(pref)
    return rows


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=SEED + 11)
    parser.add_argument("--target", type=int, default=8000)
    args = parser.parse_args()

    try:
        rng = random.Random(args.seed)
        print("Loading decontamination hashes...")
        decontam = _load_decontam_hashes()
        print(f"  decontam: {len(decontam)} hashes")

        quotas = {
            "hellaswag": 1600,
            "openbookqa": 1800,
            "winogrande": 1100,
            "sciq": 800,
            "arc_easy": 600,
            "arc_challenge": 400,
            "commonsenseqa": 1300,
            "boolq_control": 400,
        }
        scale = max(0.1, args.target / sum(quotas.values()))
        quotas = {name: max(1, int(round(value * scale))) for name, value in quotas.items()}
        loaders = {
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
                items = loaders[name]()
            except Exception as exc:
                print(f"  skipped {name}: {exc!r}")
                continue
            picked = _sample_preferences(items, quota=quota, rng=rng, decontam=decontam)
            rows.extend(picked)
            print(f"  added {len(picked)} pairs")

        rng.shuffle(rows)
        _write_jsonl(rows, args.out)

        sources = Counter(row["source"] for row in rows)
        chosen = Counter(row["chosen"].strip() for row in rows)
        rejected = Counter(row["rejected"].strip() for row in rows)
        print(f"Wrote {len(rows)} DPO preference pairs to {args.out}")
        print(f"Chosen distribution: {dict(sorted(chosen.items()))}")
        print(f"Rejected distribution: {dict(sorted(rejected.items()))}")
        print(f"Source distribution: {dict(sorted(sources.items()))}")
        return 0
    finally:
        cleanup_hf_dataset_cache()


if __name__ == "__main__":
    raise SystemExit(main())
