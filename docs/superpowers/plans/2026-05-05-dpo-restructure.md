# DPO Restructure — Continuation-Pair Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Restructure DPO to use continuation-preference pairs (chosen=correct continuation text, rejected=distractor continuation text) as the majority signal, with letter-preference pairs as a minority slice. Lower DPO β to 0.1 and run on the new SFT checkpoint produced by Plan A. Skips error-mining and UltraFeedback (deferred — out of scope for the overnight run).

**Architecture:** Add a new `_build_continuation_dpo_pair` helper and a `_run_prepare_dpo_continuation` pipeline that mirrors the existing `_run_prepare_dpo_mc_letter` flow but emits (correct continuation, distractor continuation) pairs. Add a multi-format DPO config that runs both pair types in a single training run. Tasks intentionally minimal so the pipeline can run end-to-end overnight.

**Tech Stack:** Python 3.13, uv, PyTorch, datasets, pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-05-05-sft-dpo-design.md` §3.4 — but with error mining and UltraFeedback deferred for the overnight run.

---

### Task B1: Add `preference_format` switch and continuation pair builder

**Files:**
- Modify: `configs/posttraining/dpoConfig.py` — add `preference_format` field option.
- Modify: `src/posttraining/dpo/prepare.py` — add `_build_continuation_dpo_pair`.
- Test: `tests/posttraining/test_dpo_prepare.py` — add continuation-pair tests.

- [ ] **Step 1: Verify the existing `preference_format` switch**

Read `configs/posttraining/dpoConfig.py` and find the `preference_format` field in `DPOConfig` (or `DPOSourceConfig`). It should currently accept values like `"hh_rlhf"` and `"mc_letter"` (Tasks 8 left this in place). Confirm the schema shape.

If a per-source `preference_format` enum exists, add `"continuation"` to its allowed values. If only the global `dpo.preference_format` exists, add `"continuation"` there too.

- [ ] **Step 2: Write the failing test for `_build_continuation_dpo_pair`**

Append to `tests/posttraining/test_dpo_prepare.py`:

```python
def test_build_continuation_dpo_pair_basic():
    """Build a (prompt, chosen=correct continuation, rejected=distractor) pair."""
    from src.posttraining.dpo.prepare import _build_continuation_dpo_pair
    from src.utils import build_tokenizer
    import random

    tokenizer = build_tokenizer()
    rng = random.Random(0)

    pair = _build_continuation_dpo_pair(
        user_prompt="Context: A man stands on a roof. He",
        correct_continuation="starts pulling up roofing.",
        distractor_continuations=[
            "is using wrap to wrap a pair of skis.",
            "is holding a rubik's cube.",
            "starts pulling up tomatoes.",
        ],
        tokenizer=tokenizer,
        system_prompt="You are ParrotLLM.",
        max_seq_length=1024,
        rng=rng,
    )

    assert pair is not None
    assert "prompt_tokens" in pair
    assert "chosen_tokens" in pair
    assert "rejected_tokens" in pair
    assert pair["chosen_tokens"] != pair["rejected_tokens"]
    # Chosen tokens are strictly longer than prompt (since they include the continuation).
    assert len(pair["chosen_tokens"]) > pair["prompt_len"]
    assert len(pair["rejected_tokens"]) > pair["prompt_len"]


def test_build_continuation_dpo_pair_rejects_no_distractors():
    from src.posttraining.dpo.prepare import _build_continuation_dpo_pair
    from src.utils import build_tokenizer
    import random

    tokenizer = build_tokenizer()
    rng = random.Random(0)

    pair = _build_continuation_dpo_pair(
        user_prompt="Q?",
        correct_continuation="answer",
        distractor_continuations=[],
        tokenizer=tokenizer,
        system_prompt="You are ParrotLLM.",
        max_seq_length=1024,
        rng=rng,
    )
    assert pair is None
```

- [ ] **Step 3: Run the test to confirm failure**

```bash
uv run pytest tests/posttraining/test_dpo_prepare.py::test_build_continuation_dpo_pair_basic -v
```

- [ ] **Step 4: Implement `_build_continuation_dpo_pair`**

In `src/posttraining/dpo/prepare.py`, near `_build_letter_dpo_pair`, add:

```python
def _build_continuation_dpo_pair(
    *,
    user_prompt: str,
    correct_continuation: str,
    distractor_continuations: list[str],
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Build a (prompt, chosen=correct continuation, rejected=distractor continuation) pair.

    Mirrors `pack_pair`'s output shape so the existing DPO trainer consumes it
    unchanged. The chosen response is the correct continuation; the rejected
    response is a randomly-sampled distractor.
    """
    if not distractor_continuations:
        return None
    rejected_continuation = rng.choice(distractor_continuations)

    prompt_messages = normalize_messages(
        [{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        require_final_assistant=False,
    )
    prompt_render = render_conversation(prompt_messages, add_generation_prompt=True)
    prompt_text = prompt_render.text  # ends with "\n\n### Response:\n"

    chosen_text = prompt_text + correct_continuation.strip()
    rejected_text = prompt_text + rejected_continuation.strip()

    prompt_tokens = tokenizer.encode(prompt_text)
    chosen_tokens = tokenizer.encode(chosen_text)
    rejected_tokens = tokenizer.encode(rejected_text)

    if len(chosen_tokens) > max_seq_length or len(rejected_tokens) > max_seq_length:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "prompt_len": len(prompt_tokens),
    }
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/posttraining/test_dpo_prepare.py -v
```

Expected: 7 PASS (5 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/posttraining/dpo/prepare.py tests/posttraining/test_dpo_prepare.py configs/posttraining/dpoConfig.py
git commit -m "feat(dpo): add _build_continuation_dpo_pair helper for cloze-aligned DPO"
```

---

### Task B2: Add `_run_prepare_dpo_continuation` pipeline

**Files:**
- Modify: `src/posttraining/dpo/prepare.py` — add the pipeline runner.
- Modify: `src/posttraining/dpo/prepare.py` — wire into `run_prepare_dpo` dispatcher based on `preference_format`.

For each MC source (HellaSwag, WinoGrande, OpenBookQA, ARC, SciQ, CommonsenseQA), iterate raw records, extract:
- The user_prompt (the same MC stem the SFT MC normalizers use)
- The correct continuation (the gold choice text)
- The distractor continuations (the wrong choice texts)

Then build pairs via `_build_continuation_dpo_pair`. Mirror the structure of `_run_prepare_dpo_mc_letter`.

- [ ] **Step 1: Read `_run_prepare_dpo_mc_letter`**

Read the function carefully. The pipeline:
1. Loads each source via `load_dataset`.
2. Calls `_normalize_source_record(record, source_cfg)` on each raw record.
3. The normalizer returns (messages, metadata).
4. Builds an `mc_letter` pair from the messages.

For continuation pairs, we need access to the raw choice texts BEFORE the normalizer renders them into the MC prompt. So we can't reuse `_normalize_source_record` directly — we need to extract the choice texts ourselves from the raw record schema.

For each loader, define a small extractor: `(user_prompt, correct_continuation, distractor_continuations)`.

- [ ] **Step 2: Add per-loader extractors**

In `src/posttraining/dpo/prepare.py`, add:

```python
def _extract_continuation_signals(
    record: Mapping[str, Any], loader: str
) -> tuple[str, str, list[str]] | None:
    """Extract (user_prompt, correct_continuation, distractor_continuations)
    from a raw MC record for continuation-DPO."""
    if loader == "hellaswag":
        ctx = str(record.get("ctx") or "").strip()
        endings = record.get("endings")
        label = record.get("label")
        if not ctx or not isinstance(endings, list) or len(endings) != 4 or label is None:
            return None
        try:
            gold_idx = int(label)
        except (TypeError, ValueError):
            return None
        if not (0 <= gold_idx < 4):
            return None
        cleaned = [str(e).strip() for e in endings]
        if any(not e for e in cleaned):
            return None
        # Prompt is just the context; we want the model to score natural continuations.
        return ctx, cleaned[gold_idx], [c for i, c in enumerate(cleaned) if i != gold_idx]

    if loader == "winogrande":
        sentence = str(record.get("sentence") or "").strip()
        opt1 = str(record.get("option1") or "").strip()
        opt2 = str(record.get("option2") or "").strip()
        answer = str(record.get("answer") or "").strip()
        if not sentence or not opt1 or not opt2 or answer not in {"1", "2"}:
            return None
        gold = opt1 if answer == "1" else opt2
        wrong = opt2 if answer == "1" else opt1
        # Replace the underscore in the sentence with the choice for both.
        # Continuation here is just the option that fills the blank.
        return sentence, gold, [wrong]

    if loader == "ai2_arc":
        question = str(record.get("question") or "").strip()
        raw_choices = record.get("choices") or {}
        answer_key = str(record.get("answerKey") or "").strip().upper()
        texts = raw_choices.get("text") if isinstance(raw_choices, Mapping) else None
        labels = raw_choices.get("label") if isinstance(raw_choices, Mapping) else None
        if not question or not isinstance(texts, list) or not isinstance(labels, list):
            return None
        # Find gold index by label match.
        gold_idx = None
        for i, lab in enumerate(labels):
            if str(lab).strip().upper() == answer_key:
                gold_idx = i
                break
        if gold_idx is None or not (0 <= gold_idx < len(texts)):
            return None
        cleaned = [str(t).strip() for t in texts]
        if any(not t for t in cleaned):
            return None
        return question, cleaned[gold_idx], [c for i, c in enumerate(cleaned) if i != gold_idx]

    if loader == "openbookqa":
        stem = str(record.get("question_stem") or "").strip()
        raw_choices = record.get("choices") or {}
        answer_key = str(record.get("answerKey") or "").strip().upper()
        texts = raw_choices.get("text") if isinstance(raw_choices, Mapping) else None
        labels = raw_choices.get("label") if isinstance(raw_choices, Mapping) else None
        if not stem or not isinstance(texts, list) or not isinstance(labels, list):
            return None
        gold_idx = None
        for i, lab in enumerate(labels):
            if str(lab).strip().upper() == answer_key:
                gold_idx = i
                break
        if gold_idx is None or not (0 <= gold_idx < len(texts)):
            return None
        cleaned = [str(t).strip() for t in texts]
        if any(not t for t in cleaned):
            return None
        return stem, cleaned[gold_idx], [c for i, c in enumerate(cleaned) if i != gold_idx]

    if loader == "sciq":
        q = str(record.get("question") or "").strip()
        correct = str(record.get("correct_answer") or "").strip()
        distractors = [
            str(record.get("distractor1") or "").strip(),
            str(record.get("distractor2") or "").strip(),
            str(record.get("distractor3") or "").strip(),
        ]
        if not q or not correct or any(not d for d in distractors):
            return None
        return q, correct, distractors

    if loader == "commonsense_qa":
        q = str(record.get("question") or "").strip()
        raw_choices = record.get("choices") or {}
        answer_key = str(record.get("answerKey") or "").strip().upper()
        texts = raw_choices.get("text") if isinstance(raw_choices, Mapping) else None
        labels = raw_choices.get("label") if isinstance(raw_choices, Mapping) else None
        if not q or not isinstance(texts, list) or not isinstance(labels, list):
            return None
        gold_idx = None
        for i, lab in enumerate(labels):
            if str(lab).strip().upper() == answer_key:
                gold_idx = i
                break
        if gold_idx is None or not (0 <= gold_idx < len(texts)):
            return None
        cleaned = [str(t).strip() for t in texts]
        if any(not t for t in cleaned):
            return None
        return q, cleaned[gold_idx], [c for i, c in enumerate(cleaned) if i != gold_idx]

    if loader == "mmlu":
        q = str(record.get("question") or "").strip()
        choices = record.get("choices")
        answer = record.get("answer")
        if not q or not isinstance(choices, list) or len(choices) != 4 or answer is None:
            return None
        try:
            gold_idx = int(answer)
        except (TypeError, ValueError):
            return None
        if not (0 <= gold_idx < 4):
            return None
        cleaned = [str(c).strip() for c in choices]
        if any(not c for c in cleaned):
            return None
        return q, cleaned[gold_idx], [c for i, c in enumerate(cleaned) if i != gold_idx]

    return None
```

- [ ] **Step 3: Add the pipeline runner**

```python
def _run_prepare_dpo_continuation(
    project_config: ProjectConfig,
    *,
    seed: int,
    hf_token: str | None = None,
) -> None:
    """Build (correct continuation, distractor continuation) preference pairs."""
    from datasets import load_dataset
    from src.utils import build_tokenizer

    dpo = project_config.dpo
    if dpo is None or not dpo.sources:
        raise ValueError("dpo.sources is empty")

    out_dir = Path(dpo.prepared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    tokenizer = build_tokenizer()
    decontam_index = _build_dpo_decontam_index(dpo.decontam_datasets)
    rng = random.Random(seed)

    target = sum(s.target_pairs for s in dpo.sources)
    train_target = max(0, target - dpo.dev_pairs)

    n_pairs = 0
    n_train = 0
    n_dev = 0
    n_dropped_decontam = 0
    n_dropped_invalid = 0

    with train_path.open("w", encoding="utf-8") as f_train, dev_path.open("w", encoding="utf-8") as f_dev:
        for source in dpo.sources:
            log.info("Loading source: %s (%s)", source.name, source.path)
            ds = load_dataset(
                source.path,
                source.subset,
                split=source.split,
                cache_dir=str(dpo.cache_dir) if dpo.cache_dir else None,
                token=hf_token,
            )
            for raw in ds:
                if n_pairs >= target:
                    break
                signals = _extract_continuation_signals(raw, source.loader)
                if signals is None:
                    n_dropped_invalid += 1
                    continue
                user_prompt, correct, distractors = signals
                if decontam_index.contains(user_prompt):
                    n_dropped_decontam += 1
                    continue
                packed = _build_continuation_dpo_pair(
                    user_prompt=user_prompt,
                    correct_continuation=correct,
                    distractor_continuations=distractors,
                    tokenizer=tokenizer,
                    system_prompt=dpo.system_prompt,
                    max_seq_length=dpo.max_seq_length,
                    rng=rng,
                )
                if packed is None:
                    n_dropped_invalid += 1
                    continue
                stream = f_train if n_train < train_target else f_dev
                stream.write(json.dumps(packed) + "\n")
                if stream is f_train:
                    n_train += 1
                else:
                    n_dev += 1
                n_pairs += 1
                if n_train >= train_target and n_dev >= dpo.dev_pairs:
                    break

    manifest = {
        "preference_format": "continuation",
        "n_train": n_train,
        "n_dev": n_dev,
        "n_dropped_decontam": n_dropped_decontam,
        "n_dropped_invalid": n_dropped_invalid,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("DPO continuation prepare finished: %s", manifest)
```

- [ ] **Step 4: Wire into `run_prepare_dpo` dispatcher**

Find `run_prepare_dpo` and add:

```python
    pref_format = (project_config.dpo.preference_format or "hh_rlhf").strip().lower()
    if pref_format == "continuation":
        return _run_prepare_dpo_continuation(project_config, seed=seed, hf_token=hf_token)
    if pref_format == "mc_letter":
        return _run_prepare_dpo_mc_letter(project_config, seed=seed, hf_token=hf_token)
    return _run_prepare_dpo_hh_rlhf(project_config, seed=seed, hf_token=hf_token)
```

- [ ] **Step 5: Smoke-test with a tiny config**

```bash
uv run python -c "
# Quick test: build 5 continuation pairs from HellaSwag-train without going through main.py.
from src.posttraining.dpo.prepare import _extract_continuation_signals
from datasets import load_dataset
ds = load_dataset('Rowan/hellaswag', split='train', streaming=True)
for i, rec in enumerate(ds):
    if i >= 3: break
    sig = _extract_continuation_signals(rec, 'hellaswag')
    if sig:
        prompt, correct, distractors = sig
        print(f'PROMPT: {prompt[:60]}...')
        print(f'CORRECT: {correct[:60]}')
        print(f'DISTRACTORS: {len(distractors)}')
        print('---')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/posttraining/dpo/prepare.py
git commit -m "feat(dpo): add continuation-pair preparation pipeline"
```

---

### Task B3: Build the new DPO config

**Files:**
- Create: `configs/posttraining/dpo_continuation.yaml`

- [ ] **Step 1: Create the config**

Base on `dpo_benchmark_shared.yaml`. Key differences:
- `preference_format: continuation`
- `beta: 0.1` (down from current 0.3)
- Sources: HellaSwag, WinoGrande, OpenBookQA, ARC-E, ARC-C, SciQ, CSQA — drop the loaders that aren't supported by `_extract_continuation_signals`.
- `reference_checkpoint`: best checkpoint from the new SFT run (set after SFT completes).

Concrete content (set `reference_checkpoint` to the new SFT output path):

```yaml
# (copy from dpo_benchmark_shared.yaml's outer scaffold)
dpo:
  preference_format: continuation
  beta: 0.1
  reference_checkpoint: <SET_AFTER_SFT_RUN>
  prepared_dir: data/posttraining/dpo_pairs_continuation
  runs_dir: runs/posttraining/dpo_continuation
  cache_dir: data/posttraining/hf_cache
  system_prompt: "You are ParrotLLM, a helpful assistant."
  max_seq_length: 1024
  dev_pairs: 500
  num_epochs: 1.0
  learning_rate: 2.0e-6
  warmup_ratio: 0.03
  batch_size: 4
  gradient_accumulation_steps: 1
  sources:
    - name: hellaswag
      loader: hellaswag
      path: Rowan/hellaswag
      split: train
      target_pairs: 8000
    - name: winogrande
      loader: winogrande
      path: allenai/winogrande
      subset: winogrande_xl
      split: train
      target_pairs: 5000
    - name: openbookqa
      loader: openbookqa
      path: allenai/openbookqa
      subset: main
      split: train
      target_pairs: 3000
    - name: arc_easy
      loader: ai2_arc
      path: allenai/ai2_arc
      subset: ARC-Easy
      split: train
      target_pairs: 2000
    - name: arc_challenge
      loader: ai2_arc
      path: allenai/ai2_arc
      subset: ARC-Challenge
      split: train
      target_pairs: 1000
    - name: sciq
      loader: sciq
      path: allenai/sciq
      split: train
      target_pairs: 3000
    - name: commonsense_qa
      loader: commonsense_qa
      path: tau/commonsense_qa
      split: train
      target_pairs: 3000
  decontam_datasets:
    # (copy from dpo_benchmark_shared.yaml)
```

Total target pairs: ~25k.

- [ ] **Step 2: Validate**

```bash
uv run python -c "
from configs import load_project_config
cfg = load_project_config('configs/posttraining/dpo_continuation.yaml')
print(f'pref_format: {cfg.dpo.preference_format}')
print(f'beta: {cfg.dpo.beta}')
print(f'sources: {len(cfg.dpo.sources)}')
print(f'target_pairs: {sum(s.target_pairs for s in cfg.dpo.sources)}')
"
```

- [ ] **Step 3: Commit**

```bash
git add configs/posttraining/dpo_continuation.yaml
git commit -m "config(dpo): add continuation-pair DPO recipe (beta=0.1)"
```

---

### Task B4 (run-time, not implemented as code): Run prepare + train

After Tasks B1-B3 land:

1. Update `dpo_continuation.yaml`'s `reference_checkpoint` to point at the new SFT checkpoint.
2. `uv run python main.py --stage dpo-prepare --config configs/posttraining/dpo_continuation.yaml`
3. `uv run python main.py --stage dpo --config configs/posttraining/dpo_continuation.yaml`
4. `uv run python -m leaderboard.run_benchmarks ...` against the resulting DPO checkpoint.

---

## Deferred (out of scope for overnight)

- Error mining (run SFT model first, identify wrong-answer examples, use those for DPO)
- UltraFeedback-MC integration
- β ablation across {0.1, 0.2, 0.3}
- Continuation-pair format paraphrase

These can land in Plan B-v2 once the v1 results are in.
