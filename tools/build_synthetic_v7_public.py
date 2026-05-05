"""v7 public-data builder — broad mix of English MC TRAIN splits.

Reformats train splits from multiple datasets in the same task family
as the visible/hidden leaderboard benches into the raw MC template
("Context: ...\\nA) ...\\nAnswer:" + " <letter>"). The DIVERSITY of
sources is the point: the model should learn the underlying skill of
"do MC reasoning given context + 4 options" without overfitting to any
single dataset's quirks.

Sources (all TRAIN splits — disjoint from validation/test by construction):

  - HellaSwag-train (~39k available)         narrative continuation
  - OpenBookQA-train (~5k)                   elementary science MC
  - WinoGrande-train (~40k)                  pronoun resolution
  - SciQ-train (~12k)                        science MC
  - ARC-Easy-train (~2k)                     grade-school science
  - ARC-Challenge-train (~1k)                challenging grade-school science
  - CommonsenseQA-train (~10k)               commonsense MC

Quotas chosen to total ~5500 rows. Decontam against every known visible
+ extended test/validation split is handled at training time by
build_sft_datasets via DECONTAM_LOADERS. Local pre-filter as
belt-and-suspenders.

Output: data/synthetic/sft_v7_public.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

OUT_PATH = Path("data/synthetic/sft_v7_public.jsonl")
SEED = 44
TARGET_TOTAL = 5500


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
        except Exception as e:
            print(f"  decontam: SKIP {name} ({e!r})")
    idx = build_decontam_index(texts)
    print(f"  decontam: {len(idx)} unique hashes")
    return idx


def _make_instruction(stem: str, choices: list[str], answer_idx: int) -> dict | None:
    if not stem or len(choices) < 2 or answer_idx >= len(choices):
        return None
    if any(not c for c in choices):
        return None
    if len(choices) > 4:
        keep = [answer_idx]
        candidates = [i for i in range(len(choices)) if i != answer_idx]
        random.shuffle(candidates)
        keep.extend(candidates[:3])
        keep.sort()
        new_idx = keep.index(answer_idx)
        choices = [choices[i] for i in keep]
        answer_idx = new_idx
    letters = "ABCDE"[: len(choices)]
    block = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    instruction = f"Context: {stem.strip()}\n{block}\nAnswer:"
    return {"instruction": instruction, "response": f" {letters[answer_idx]}"}


def _is_clean(ex: dict, decontam: set[str]) -> bool:
    blob = ex["instruction"] + " " + ex["response"]
    return _sha1(_normalise_for_hash(blob)) not in decontam


def pull_hellaswag(rng, n, decontam):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="train")
    out, rows = [], list(ds)
    rng.shuffle(rows)
    for r in rows:
        ctx = (r.get("ctx_a", "") + " " + r.get("ctx_b", "")).strip()
        endings = list(r.get("endings", []))
        try:
            ans = int(r.get("label", -1))
        except Exception:
            continue
        ex = _make_instruction(ctx, endings, ans)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_obqa(rng, n, decontam):
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    out, rows = [], list(ds)
    rng.shuffle(rows)
    for r in rows:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        ans = r["answerKey"]
        if ans not in labels:
            continue
        ans_idx = labels.index(ans)
        ex = _make_instruction(r["question_stem"], texts, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_winogrande(rng, n, decontam):
    from datasets import load_dataset
    ds = load_dataset(
        "allenai/winogrande", "winogrande_xl",
        split="train", trust_remote_code=True,
    )
    out, rows = [], list(ds)
    rng.shuffle(rows)
    for r in rows:
        sentence = r["sentence"]
        opt1, opt2 = r["option1"], r["option2"]
        ans = str(r.get("answer", "")).strip()
        if ans not in {"1", "2"}:
            continue
        ans_idx = int(ans) - 1
        ex = _make_instruction(sentence, [opt1, opt2], ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_sciq(rng, n, decontam):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    out, rows = [], list(ds)
    rng.shuffle(rows)
    for r in rows:
        choices = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        choices = [choices[i] for i in order]
        ans_idx = order.index(0)
        ex = _make_instruction(r["question"], choices, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_arc(rng, n, decontam, cfg):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", cfg, split="train")
    out, rows = [], list(ds)
    rng.shuffle(rows)
    for r in rows:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        ans = r["answerKey"]
        if ans not in labels:
            continue
        ans_idx = labels.index(ans)
        ex = _make_instruction(r["question"], texts, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_csqa(rng, n, decontam):
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="train")
    out, rows = [], list(ds)
    rng.shuffle(rows)
    for r in rows:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        ans = r["answerKey"]
        if not ans or ans not in labels:
            continue
        ans_idx = labels.index(ans)
        ex = _make_instruction(r["question"], texts, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET_TOTAL)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    rng = random.Random(SEED)
    print("Loading decontam index …")
    decontam = _load_decontam_hashes()

    quotas = {
        "hellaswag": 1500,
        "obqa":      1000,
        "winogrande": 1000,
        "sciq":       450,
        "arc_easy":   300,
        "arc_chall":  150,
        "csqa":       1100,
    }
    scale = args.target / sum(quotas.values())
    quotas = {k: max(1, int(round(v * scale))) for k, v in quotas.items()}
    print(f"Quotas: {quotas}")

    rows: list[dict] = []
    print("Pulling HellaSwag-train …")
    rows += pull_hellaswag(rng, quotas["hellaswag"], decontam)
    print(f"  +{len(rows)} cumulative")
    print("Pulling OpenBookQA-train …")
    rows += pull_obqa(rng, quotas["obqa"], decontam)
    print(f"  +{len(rows)} cumulative")
    print("Pulling WinoGrande-train …")
    try:
        rows += pull_winogrande(rng, quotas["winogrande"], decontam)
    except Exception as e:
        print(f"  WinoGrande failed: {e!r} — skipping")
    print(f"  +{len(rows)} cumulative")
    print("Pulling SciQ-train …")
    rows += pull_sciq(rng, quotas["sciq"], decontam)
    print(f"  +{len(rows)} cumulative")
    print("Pulling ARC-Easy-train …")
    rows += pull_arc(rng, quotas["arc_easy"], decontam, "ARC-Easy")
    print(f"  +{len(rows)} cumulative")
    print("Pulling ARC-Challenge-train …")
    rows += pull_arc(rng, quotas["arc_chall"], decontam, "ARC-Challenge")
    print(f"  +{len(rows)} cumulative")
    print("Pulling CommonsenseQA-train …")
    rows += pull_csqa(rng, quotas["csqa"], decontam)
    print(f"  +{len(rows)} cumulative")

    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for ex in rows:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    letters = Counter(ex["response"].strip() for ex in rows)
    print(f"Wrote {len(rows)} examples to {args.out}")
    print(f"Letter distribution: {dict(letters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
