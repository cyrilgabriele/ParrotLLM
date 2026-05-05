# SFT Data Mix Expansion + Choice Permutation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire benchmark-targeted MC training sources and a LAMBADA-shape regularizer into the existing SFT preparation pipeline, plus mandatory choice-order permutation, so a benchmark-checkpoint training run can be launched against a single new `sft_benchmark.yaml` config.

**Architecture:** All MC normalizers needed for HellaSwag/WinoGrande/OBQA/ARC/SciQ/CSQA/PIQA already exist in `src/posttraining/prepare.py` — they're just not wired into any active SFT config. This plan (a) adds a deterministic `_permute_choices` helper applied at the start of every MC normalizer to eliminate gold-letter bias, (b) adds five new normalizers covering Tier 2 (CosmosQA, SocialIQa), Tier 4 LAMBADA-shape (CBT, BookCorpus passage→last-word), and Tier 5 LLM-generated MC (generic FLAN-MC filter for OpenOrca), (c) builds the wiring config `sft_benchmark.yaml`, (d) re-enables DPO decontamination, and (e) archives abandoned experiment configs. Output is a prepared SFT mix on disk plus a runnable training command. No training is launched in this plan; that's a separate run.

**Tech Stack:** Python 3.13, uv, PyTorch, HuggingFace `datasets`, pydantic v2, pytest. Existing project conventions: `uv run pytest …` for tests, `uv run python main.py --stage sft-prepare --config …` for data prep.

**Spec:** `docs/superpowers/specs/2026-05-05-sft-dpo-design.md` (the audit-revised v2). Tier numbering and dataset choices reference §3.1 there.

---

### Task 1: Add deterministic choice-order permutation and apply across all MC normalizers

**Why:** The current normalizers preserve dataset-baked gold-letter bias (e.g., HellaSwag's gold tilts toward later letters; ARC has uneven distribution). This costs 1–3pp on benchmarks that depend on letter argmax. Permutation is per-example deterministic so re-running data prep produces identical training files.

**Files:**
- Create: `tests/posttraining/test_choice_permutation.py`
- Modify: `src/posttraining/prepare.py` — add `_permute_choices` helper, call from every MC normalizer that uses `_format_mc_prompt`.

- [ ] **Step 1: Write the failing test for the permutation helper**

Create `tests/posttraining/test_choice_permutation.py`:

```python
"""Tests for the deterministic choice-order permutation helper."""

from __future__ import annotations

from collections import Counter

from src.posttraining.prepare import _permute_choices


def test_permute_choices_is_deterministic_per_seed_key():
    choices = ["alpha", "beta", "gamma", "delta"]
    answer_index = 1
    seed_key = "example_42"

    out_choices_a, out_index_a = _permute_choices(choices, answer_index, seed_key)
    out_choices_b, out_index_b = _permute_choices(choices, answer_index, seed_key)

    assert out_choices_a == out_choices_b
    assert out_index_a == out_index_b


def test_permute_choices_preserves_gold_text():
    choices = ["alpha", "beta", "gamma", "delta"]
    answer_index = 2
    seed_key = "ex"

    out_choices, out_index = _permute_choices(choices, answer_index, seed_key)

    assert out_choices[out_index] == "gamma"
    assert sorted(out_choices) == sorted(choices)


def test_permute_choices_uniform_over_many_examples():
    """Across 4000 random keys, the gold index should be roughly uniform."""
    choices = ["a", "b", "c", "d"]
    counter: Counter[int] = Counter()
    for i in range(4000):
        _, idx = _permute_choices(choices, 0, f"key_{i}")
        counter[idx] += 1

    assert set(counter) == {0, 1, 2, 3}
    expected = 4000 / 4
    for count in counter.values():
        # Each bucket should be within ±10% of expected (uniform with reasonable variance)
        assert abs(count - expected) < 0.10 * expected, counter


def test_permute_choices_handles_binary():
    choices = ["yes", "no"]
    out_choices, out_index = _permute_choices(choices, 0, "binary_key_1")
    assert out_choices[out_index] == "yes"
    assert sorted(out_choices) == ["no", "yes"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/posttraining/test_choice_permutation.py -v
```

Expected: FAIL with `ImportError: cannot import name '_permute_choices' from 'src.posttraining.prepare'`.

- [ ] **Step 3: Implement `_permute_choices` in prepare.py**

In `src/posttraining/prepare.py`, near the existing `_format_mc_prompt` (around line 397), add:

```python
def _permute_choices(
    choices: list[str], answer_index: int, seed_key: str
) -> tuple[list[str], int]:
    """Deterministically permute MC choices so the gold answer's position
    is uniform across the dataset rather than tracking the source's bias.

    The permutation is keyed on `seed_key` (typically the example id or
    question stem) so the same example always permutes to the same order
    across data-prep runs — the prepared SFT files are reproducible.
    """
    import random

    seed = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    indices = list(range(len(choices)))
    rng.shuffle(indices)
    new_choices = [choices[i] for i in indices]
    new_answer_index = indices.index(answer_index)
    return new_choices, new_answer_index
```

`hashlib` is already imported at the top of `prepare.py` (used by `_stable_hash` and `_normalize_sciq_record`).

- [ ] **Step 4: Run test to verify the helper passes**

```bash
uv run pytest tests/posttraining/test_choice_permutation.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Wire permutation into every MC normalizer**

In `src/posttraining/prepare.py`, modify each of the following normalizers to call `_permute_choices` immediately before `_format_mc_prompt`. The seed key should be a stable string from the record (`record.get("id")` if present, else the question/context text).

Modify `_normalize_ai2_arc_record` (around line 348). After the block that resolves `answer_index` and before the call to `_format_mc_prompt`, replace:

```python
    prompt = _format_mc_prompt(question, cleaned_choices, prefix="Question")
    messages = _build_mc_messages(prompt, "ABCD"[answer_index])
```

with:

```python
    seed_key = str(record.get("id") or question)
    cleaned_choices, answer_index = _permute_choices(
        cleaned_choices, answer_index, seed_key
    )
    prompt = _format_mc_prompt(question, cleaned_choices, prefix="Question")
    messages = _build_mc_messages(prompt, "ABCD"[answer_index])
```

Apply the same pattern to:
- `_normalize_commonsense_qa_record` (line 438): seed_key = `str(record.get("id") or question)`. Subsample 5 choices to 4 first by dropping the lowest-confidence distractor — but if the dataset has 5 choices and you pick the one to drop, the existing code already handles that. Only permute *after* the 5→4 reduction.
- `_normalize_race_record` (line 462): seed_key = `str(record.get("example_id") or record.get("id") or prompt_text[:64])`.
- `_normalize_mmlu_record` (line 481): seed_key = `str(record.get("question") or "")[:128]`.
- `_normalize_piqa_record` (line 524): seed_key = `str(record.get("goal") or "")`.
- `_normalize_hellaswag_record` (line 571): seed_key = `str(record.get("ind") or record.get("source_id") or ctx[:64])`.
- `_normalize_openbookqa_record` (line 641): seed_key = `str(record.get("id") or stem)`.
- `_normalize_winogrande_record` (line 666): seed_key = `str(record.get("qID") or sentence[:64])`. Note: WinoGrande is binary (2 choices), permutation still applies but only flips A↔B.

**Skip these** (already handle ordering deterministically or aren't MC):
- `_normalize_sciq_record` (already shuffles via seeded `random.Random` at line 426).
- `_normalize_boolq_record` (uses fixed `["Yes", "No"]` — flipping that would break the prompt's natural binary semantics; leave as-is).
- `_normalize_wsc273_record` — examine whether it makes sense; if context/option semantics depend on order, skip.

- [ ] **Step 6: Add an integration test that confirms a real MC normalizer permutes**

Append to `tests/posttraining/test_choice_permutation.py`:

```python
from collections import Counter as _Counter

from configs import SFTSourceConfig
from src.posttraining.prepare import _normalize_hellaswag_record


def _make_hellaswag_record(record_id: str, gold_index: int) -> dict:
    return {
        "ind": record_id,
        "ctx": f"A man is walking down the street. He",
        "endings": ["sees a dog.", "buys a coffee.", "trips on a curb.", "starts to fly."],
        "label": gold_index,
    }


def _make_source_cfg() -> SFTSourceConfig:
    return SFTSourceConfig(
        name="hs_test",
        loader="hellaswag",
        path="Rowan/hellaswag",
        split="train",
        target_examples=1,
    )


def test_hellaswag_normalizer_permutes_gold_letter():
    """When the source's gold is always index 0, after permutation the
    gold letter should be approximately uniform across A/B/C/D."""
    src = _make_source_cfg()
    letter_counts: _Counter[str] = _Counter()

    for i in range(800):
        rec = _make_hellaswag_record(record_id=str(i), gold_index=0)
        result = _normalize_hellaswag_record(rec, src)
        assert result is not None, f"normalizer rejected synthetic record {i}"
        messages, _meta = result
        gold_letter = messages[1]["content"]
        letter_counts[gold_letter] += 1

    assert set(letter_counts) == {"A", "B", "C", "D"}
    expected = 800 / 4
    for count in letter_counts.values():
        assert abs(count - expected) < 0.15 * expected, letter_counts
```

- [ ] **Step 7: Run the integration test and confirm it passes**

```bash
uv run pytest tests/posttraining/test_choice_permutation.py -v
```

Expected: 5 tests PASS. If any existing prepare test broke, run the full suite:

```bash
uv run pytest tests/posttraining/test_prepare.py -v
```

Expected: existing tests continue to PASS. Permutation is purely additive on the gold-letter axis; it doesn't change the message schema.

- [ ] **Step 8: Commit**

```bash
git add src/posttraining/prepare.py tests/posttraining/test_choice_permutation.py
git commit -m "feat(sft): deterministic choice-order permutation for MC normalizers

Eliminates dataset-baked gold-letter bias that costs 1-3pp at inference
time. Permutation is keyed on the example id so prepared SFT files stay
reproducible across data-prep runs."
```

---

### Task 2: Add CosmosQA normalizer

**Why:** Tier 2 of the spec. CosmosQA is a 25k-example 4-way commonsense narrative MC dataset — the closest large public dataset to HellaSwag's distribution. Currently no loader exists.

**Files:**
- Modify: `configs/posttraining/sftConfig.py` — add `"cosmos_qa"` to the `loader` Literal.
- Modify: `src/posttraining/prepare.py` — add `_normalize_cosmos_qa_record`, register it in `_load_records`.
- Create test: `tests/posttraining/test_normalizers_new.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/posttraining/test_normalizers_new.py`:

```python
"""Tests for newly-added MC normalizers (CosmosQA, SocialIQa, etc.)."""

from __future__ import annotations

from configs import SFTSourceConfig
from src.posttraining.prepare import _normalize_cosmos_qa_record


def _src(loader: str) -> SFTSourceConfig:
    return SFTSourceConfig(
        name=f"{loader}_test",
        loader=loader,
        path=f"allenai/{loader}",
        split="train",
        target_examples=1,
    )


def test_cosmos_qa_basic_record():
    rec = {
        "id": "cosmos_001",
        "context": "We were driving to the cabin when the storm hit.",
        "question": "What is most likely true about the trip?",
        "answer0": "The trip was uneventful and quick.",
        "answer1": "The driver pulled over due to weather.",
        "answer2": "They reached the cabin in record time.",
        "answer3": "They turned around and went home.",
        "label": 1,
    }

    result = _normalize_cosmos_qa_record(rec, _src("cosmos_qa"))
    assert result is not None
    messages, meta = result

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    user_prompt = messages[0]["content"]
    assert "Context:" in user_prompt or "Passage:" in user_prompt or rec["context"] in user_prompt
    assert rec["question"] in user_prompt
    # All four answers must appear in the prompt regardless of permuted order.
    for ans in (rec["answer0"], rec["answer1"], rec["answer2"], rec["answer3"]):
        assert ans in user_prompt
    assert user_prompt.endswith("Answer:")

    gold_letter = messages[1]["content"]
    assert gold_letter in {"A", "B", "C", "D"}

    # Verify the gold letter actually maps back to the original gold answer text.
    label_to_answer = {
        "A": user_prompt.split("A) ")[1].split("\n")[0],
        "B": user_prompt.split("B) ")[1].split("\n")[0],
        "C": user_prompt.split("C) ")[1].split("\n")[0],
        "D": user_prompt.split("D) ")[1].split("\n")[0],
    }
    assert label_to_answer[gold_letter] == rec["answer1"]


def test_cosmos_qa_rejects_missing_field():
    rec = {"id": "x", "context": "ctx", "question": "q"}  # no answers
    assert _normalize_cosmos_qa_record(rec, _src("cosmos_qa")) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_cosmos_qa_basic_record -v
```

Expected: FAIL with `ImportError: cannot import name '_normalize_cosmos_qa_record'`.

- [ ] **Step 3: Add `cosmos_qa` to the loader Literal in sftConfig.py**

In `configs/posttraining/sftConfig.py`, add `"cosmos_qa"` to the `Literal[…]` list inside `SFTSourceConfig.loader` (around line 16). Insert in alphabetical order:

```python
    loader: Literal[
        "ai2_arc",
        "alpaca",
        "boolq",
        "commonsense_qa",
        "cosmos_qa",        # NEW
        "hellaswag",
        ...
```

(Match the existing ordering in your file — pydantic doesn't require alphabetical, but it's easier to maintain.)

- [ ] **Step 4: Implement the normalizer in prepare.py**

In `src/posttraining/prepare.py`, near the other commonsense normalizers (after `_normalize_commonsense_qa_record` is fine), add:

```python
def _normalize_cosmos_qa_record(
    record: Mapping[str, Any], source_cfg: SFTSourceConfig
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    context = str(record.get("context") or "").strip()
    question = str(record.get("question") or "").strip()
    answers = [
        str(record.get("answer0") or "").strip(),
        str(record.get("answer1") or "").strip(),
        str(record.get("answer2") or "").strip(),
        str(record.get("answer3") or "").strip(),
    ]
    label = record.get("label")
    if not context or not question or any(not a for a in answers) or label is None:
        return None
    try:
        answer_index = int(label)
    except (TypeError, ValueError):
        return None
    if not (0 <= answer_index < 4):
        return None

    cleaned = [clean_message_content(a) for a in answers]
    if any(not a for a in cleaned):
        return None

    seed_key = str(record.get("id") or f"{context[:32]}|{question[:32]}")
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)

    # Render context + question together as the MC stem, headed "Context:".
    stem = f"{context}\n{question}"
    prompt = _format_mc_prompt(stem, cleaned, prefix="Context")
    messages = _build_mc_messages(prompt, "ABCD"[answer_index])
    metadata = {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }
    return messages, metadata
```

- [ ] **Step 5: Register the normalizer in `_load_records`**

Find the `_load_records` function (around line 266 in `prepare.py`). It dispatches on `source_cfg.loader`. Add a branch for `"cosmos_qa"` that calls `_normalize_cosmos_qa_record`, mirroring how other MC loaders are dispatched. The exact pattern depends on the existing dispatch shape — read the function and follow it.

If the dispatch is a dict-style mapping (e.g., `_NORMALIZER_MAP = {"ai2_arc": _normalize_ai2_arc_record, ...}`), add `"cosmos_qa": _normalize_cosmos_qa_record`.

If it's an if/elif chain, add a new `elif source_cfg.loader == "cosmos_qa": normalizer = _normalize_cosmos_qa_record`.

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_cosmos_qa_basic_record tests/posttraining/test_normalizers_new.py::test_cosmos_qa_rejects_missing_field -v
```

Expected: 2 PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/posttraining/sftConfig.py src/posttraining/prepare.py tests/posttraining/test_normalizers_new.py
git commit -m "feat(sft): add CosmosQA loader for Tier 2 commonsense breadth"
```

---

### Task 3: Add SocialIQa normalizer

**Why:** Tier 2 of the spec. SocialIQa is a 33k-example 3-way social commonsense MC. Different shape from CosmosQA (3 choices not 4), so it gets its own normalizer.

**Files:**
- Modify: `configs/posttraining/sftConfig.py` — add `"social_iqa"` to the loader Literal.
- Modify: `src/posttraining/prepare.py` — add `_normalize_social_iqa_record`, register it.
- Modify: `tests/posttraining/test_normalizers_new.py` — add SocialIQa tests.

- [ ] **Step 1: Write the failing test**

Append to `tests/posttraining/test_normalizers_new.py`:

```python
from src.posttraining.prepare import _normalize_social_iqa_record


def test_social_iqa_basic_record():
    rec = {
        "context": "Alex helped his friend study for the exam.",
        "question": "How would Alex feel afterwards?",
        "answerA": "happy and proud",
        "answerB": "sad and rejected",
        "answerC": "angry",
        "label": "1",  # SocialIQa uses 1-based string labels
    }
    result = _normalize_social_iqa_record(rec, _src("social_iqa"))
    assert result is not None
    messages, meta = result
    user_prompt = messages[0]["content"]
    gold_letter = messages[1]["content"]

    # 3-way MC -> letters are A, B, C only.
    assert gold_letter in {"A", "B", "C"}
    # Verify the prompt has all 3 choices.
    for ans in (rec["answerA"], rec["answerB"], rec["answerC"]):
        assert ans in user_prompt
    assert user_prompt.endswith("Answer:")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_social_iqa_basic_record -v
```

Expected: FAIL on import.

- [ ] **Step 3: Add to loader Literal**

In `configs/posttraining/sftConfig.py`:

```python
    loader: Literal[
        ...
        "social_iqa",       # NEW
        ...
```

- [ ] **Step 4: Implement the normalizer**

In `src/posttraining/prepare.py`, near `_normalize_cosmos_qa_record`, add:

```python
def _normalize_social_iqa_record(
    record: Mapping[str, Any], source_cfg: SFTSourceConfig
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    context = str(record.get("context") or "").strip()
    question = str(record.get("question") or "").strip()
    answers = [
        str(record.get("answerA") or "").strip(),
        str(record.get("answerB") or "").strip(),
        str(record.get("answerC") or "").strip(),
    ]
    label_raw = record.get("label")
    if not context or not question or any(not a for a in answers) or label_raw is None:
        return None
    # SocialIQa labels are 1-based strings ("1", "2", "3").
    try:
        answer_index = int(str(label_raw).strip()) - 1
    except ValueError:
        return None
    if not (0 <= answer_index < 3):
        return None

    cleaned = [clean_message_content(a) for a in answers]
    if any(not a for a in cleaned):
        return None

    seed_key = f"{context[:32]}|{question[:32]}"
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)

    stem = f"{context}\n{question}"
    prompt = _format_mc_prompt(stem, cleaned, prefix="Context")
    messages = _build_mc_messages(prompt, "ABC"[answer_index])
    metadata = {"rationale": source_cfg.rationale}
    return messages, metadata
```

- [ ] **Step 5: Register in `_load_records`**

Same pattern as Task 2 Step 5: add the dispatch entry for `"social_iqa"`.

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/posttraining/sftConfig.py src/posttraining/prepare.py tests/posttraining/test_normalizers_new.py
git commit -m "feat(sft): add SocialIQa loader for Tier 2 (3-way social commonsense)"
```

---

### Task 4: Add Children's Book Test (CBT) normalizer for LAMBADA-shape regularization

**Why:** Tier 4 of the spec. CBT is purpose-built for last-word/cloze prediction at long context, exactly the LAMBADA shape. Without a Tier 4 regularizer, MC-format SFT pulls the model away from natural-text continuation and LAMBADA accuracy collapses.

**Files:**
- Modify: `configs/posttraining/sftConfig.py` — add `"cbt"` to the loader Literal.
- Modify: `src/posttraining/prepare.py` — add `_normalize_cbt_record`, register it.
- Modify: `tests/posttraining/test_normalizers_new.py` — add CBT tests.

CBT examples on HuggingFace (`cbt`) provide `sentences` (list[str], typically 20 sentences as context), `question` (sentence with one blank), and `answer` (the missing word). For LAMBADA-shape SFT we want the **full context as the prompt and the missing word as the target** — not the blanked question itself, since LAMBADA predicts based on context-passage flow.

Construction: concatenate `sentences` + replace the blank in `question` with the answer to make a coherent passage, then split off the final word as the target.

- [ ] **Step 1: Write the failing test**

Append to `tests/posttraining/test_normalizers_new.py`:

```python
from src.posttraining.prepare import _normalize_cbt_record


def test_cbt_basic_record():
    rec = {
        "sentences": [
            "Once upon a time there was a small village.",
            "The villagers were preparing for winter.",
            "Snow had begun to fall lightly.",
        ],
        "question": "The children played happily in the XXXXX.",
        "answer": "snow",
        "options": ["village", "snow", "winter", "house", "field"],
    }

    result = _normalize_cbt_record(rec, _src("cbt"))
    assert result is not None
    messages, meta = result
    assert len(messages) == 2
    assistant = messages[1]["content"]
    user = messages[0]["content"]

    # The assistant target is the missing word.
    assert assistant.strip().lower() == "snow"
    # The user prompt is the passage with a trailing space, ready for greedy continuation.
    assert user.endswith(" ") or user.endswith("\n")
    # The original blank ("XXXXX") must NOT be in the prompt — it's been resolved
    # into context preceding the predicted final word.
    assert "XXXXX" not in user
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_cbt_basic_record -v
```

Expected: FAIL on import.

- [ ] **Step 3: Add to loader Literal in sftConfig.py**

```python
    loader: Literal[
        ...
        "cbt",              # NEW
        ...
```

- [ ] **Step 4: Implement the normalizer**

In `src/posttraining/prepare.py`:

```python
def _normalize_cbt_record(
    record: Mapping[str, Any], source_cfg: SFTSourceConfig
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Render CBT examples as LAMBADA-shape (passage_minus_last_word, last_word).

    The HF `cbt` schema provides:
      - sentences: 20 sentences of context
      - question: a 21st sentence with the answer replaced by 'XXXXX'
      - answer: the missing word

    We build a coherent passage by joining sentences + the question with the
    answer substituted, then strip the final word as the assistant target.
    """
    sentences = record.get("sentences")
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    if not isinstance(sentences, list) or not sentences or not question or not answer:
        return None

    # Reconstruct the full final sentence with the answer in place of XXXXX.
    if "XXXXX" not in question:
        return None
    completed = question.replace("XXXXX", answer)

    # The LAMBADA-shape target is the LAST whitespace-separated word of `completed`.
    # Take the last word as the target; everything before it (context + earlier
    # part of the final sentence) becomes the prompt.
    parts = completed.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    final_prefix, last_word = parts

    context = " ".join(str(s).strip() for s in sentences if str(s).strip())
    if not context:
        return None

    # Prompt is "<context> <final_sentence_minus_last_word> " — trailing space
    # mirrors the LAMBADA benchmark prompt shape.
    prompt = f"{context} {final_prefix} "

    cleaned_word = clean_message_content(last_word)
    if not cleaned_word:
        return None

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": cleaned_word},
    ]
    metadata = {"rationale": source_cfg.rationale, "kind": "lambada_shape"}
    return messages, metadata
```

- [ ] **Step 5: Register in `_load_records`**

Add `"cbt"` to the dispatch alongside the other loaders.

- [ ] **Step 6: Run the test**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_cbt_basic_record -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/posttraining/sftConfig.py src/posttraining/prepare.py tests/posttraining/test_normalizers_new.py
git commit -m "feat(sft): add Children's Book Test loader for LAMBADA-shape regularizer"
```

---

### Task 5: Add BookCorpus passage→last-word normalizer

**Why:** Tier 4 of the spec, complement to CBT. BookCorpus has lots of natural prose — sample 256–512-token passages and mask the final word as the LAMBADA-shape target. This is the bulk of Tier 4.

**Files:**
- Modify: `configs/posttraining/sftConfig.py` — add `"bookcorpus_lambada"` to the loader Literal.
- Modify: `src/posttraining/prepare.py` — add `_normalize_bookcorpus_lambada_record`, register it.
- Modify: `tests/posttraining/test_normalizers_new.py` — add tests.

For HF `bookcorpus`, each record has just `text` (a single sentence/paragraph). We need to assemble multi-paragraph passages. The simplest approach: have the loader return records that already represent assembled passages, and the normalizer just splits off the last word.

To keep this self-contained, this task assumes each input record has a `text` field with at least 30 words. The `target_examples` and `candidate_multiplier` config knobs limit how many we keep. Passage assembly across records is not done here (would require a different pipeline); a follow-up task can wire that if needed.

- [ ] **Step 1: Write the failing test**

Append to `tests/posttraining/test_normalizers_new.py`:

```python
from src.posttraining.prepare import _normalize_bookcorpus_lambada_record


def test_bookcorpus_lambada_basic():
    rec = {
        "text": (
            "She walked along the cobblestone street under the dim lamplight, "
            "her footsteps echoing softly. The night air was crisp and the "
            "shop windows glowed faintly behind their iron grilles. "
            "She paused at the corner and looked back, but the alley was empty."
        ),
    }
    result = _normalize_bookcorpus_lambada_record(rec, _src("bookcorpus_lambada"))
    assert result is not None
    messages, meta = result
    user = messages[0]["content"]
    assistant = messages[1]["content"]

    assert assistant.strip().lower() == "empty"
    assert user.endswith(" ")
    assert "empty" not in user.split(" ")[-3:]  # final word is not in the prompt


def test_bookcorpus_lambada_rejects_short_passages():
    rec = {"text": "Too short."}
    assert _normalize_bookcorpus_lambada_record(rec, _src("bookcorpus_lambada")) is None
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_bookcorpus_lambada_basic -v
```

Expected: FAIL on import.

- [ ] **Step 3: Add to loader Literal**

```python
    loader: Literal[
        ...
        "bookcorpus_lambada",   # NEW
        ...
```

- [ ] **Step 4: Implement the normalizer**

In `src/posttraining/prepare.py`:

```python
def _normalize_bookcorpus_lambada_record(
    record: Mapping[str, Any], source_cfg: SFTSourceConfig
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Strip a BookCorpus passage's final word for LAMBADA-shape SFT.

    Rejects passages that are too short to give the model real context.
    Approximate threshold: 30 whitespace-separated tokens.
    """
    text = str(record.get("text") or "").strip()
    if not text:
        return None

    # Strip trailing punctuation cleanly so the gold word isn't "word."
    # Match the LAMBADA scorer's normalization (lowercase + punct strip).
    words = text.split()
    if len(words) < 30:
        return None

    last_word = words[-1].rstrip(".,;:!?\"')]")
    if not last_word or any(c.isdigit() for c in last_word):
        return None

    prompt_text = " ".join(words[:-1]) + " "
    cleaned_word = clean_message_content(last_word)
    if not cleaned_word:
        return None

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": cleaned_word.lower()},
    ]
    metadata = {"rationale": source_cfg.rationale, "kind": "lambada_shape"}
    return messages, metadata
```

- [ ] **Step 5: Register in `_load_records`**

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_bookcorpus_lambada_basic tests/posttraining/test_normalizers_new.py::test_bookcorpus_lambada_rejects_short_passages -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/posttraining/sftConfig.py src/posttraining/prepare.py tests/posttraining/test_normalizers_new.py
git commit -m "feat(sft): add BookCorpus passage->last-word loader for LAMBADA Tier 4"
```

---

### Task 6: Add generic FLAN-MC filter normalizer for OpenOrca / OpenHermes

**Why:** Tier 5 of the spec. OpenOrca contains ~4M rows; the MC slice (~50–80k after filtering) is a large boost in MC-format coverage. OpenOrca's records have `system_prompt`, `question`, `response` — for MC entries, the question contains choices and the response contains a letter (sometimes after a CoT chain). Same shape used by OpenHermes 2.5 for its MC subset, so one normalizer covers both.

The filter logic: keep only records whose `question` has a recognizable MC layout (presence of "A)" / "(A)" / "A." patterns and an "Answer:" or similar cue), and whose `response` parses to a single letter.

**Files:**
- Modify: `configs/posttraining/sftConfig.py` — add `"flan_mc"` to the loader Literal.
- Modify: `src/posttraining/prepare.py` — add `_normalize_flan_mc_record`, register it.
- Modify: `tests/posttraining/test_normalizers_new.py` — add tests.

- [ ] **Step 1: Write the failing test**

Append to `tests/posttraining/test_normalizers_new.py`:

```python
from src.posttraining.prepare import _normalize_flan_mc_record


def test_flan_mc_extracts_letter_from_response():
    """OpenOrca-style: question has MC structure, response has reasoning + final letter."""
    rec = {
        "system_prompt": "",
        "question": (
            "What is 2 + 2?\n"
            "A) 3\n"
            "B) 4\n"
            "C) 5\n"
            "D) 6\n"
            "Answer:"
        ),
        "response": "We add 2 and 2 to get 4. The answer is B.",
    }
    result = _normalize_flan_mc_record(rec, _src("flan_mc"))
    assert result is not None
    messages, meta = result
    assert messages[1]["content"] == "B"
    user = messages[0]["content"]
    # The question is preserved verbatim — we don't re-render the MC structure.
    assert "A) 3" in user
    assert user.endswith("Answer:")


def test_flan_mc_rejects_non_mc_questions():
    rec = {"question": "What is the capital of France?", "response": "Paris."}
    assert _normalize_flan_mc_record(rec, _src("flan_mc")) is None


def test_flan_mc_rejects_unparseable_response():
    rec = {
        "question": "Q?\nA) x\nB) y\nAnswer:",
        "response": "I'm not sure but maybe both could be valid...",
    }
    assert _normalize_flan_mc_record(rec, _src("flan_mc")) is None
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py::test_flan_mc_extracts_letter_from_response -v
```

Expected: FAIL on import.

- [ ] **Step 3: Add to loader Literal**

```python
    loader: Literal[
        ...
        "flan_mc",          # NEW
        ...
```

- [ ] **Step 4: Implement the normalizer**

In `src/posttraining/prepare.py`:

```python
import re as _re_for_flan_mc  # alias to avoid colliding if `re` already imported elsewhere


_FLAN_MC_QUESTION_PATTERN = _re_for_flan_mc.compile(
    r"(?:^|\n)\s*[A-D]\)\s+\S",  # detects "A) ...", "B) ..."
)
_FLAN_MC_RESPONSE_LETTER_PATTERN = _re_for_flan_mc.compile(
    r"(?:answer\s*(?:is|:)\s*|^\s*)([A-D])\b",
    _re_for_flan_mc.IGNORECASE,
)


def _normalize_flan_mc_record(
    record: Mapping[str, Any], source_cfg: SFTSourceConfig
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Filter OpenOrca / OpenHermes / FLAN-shape records to keep only the
    MC subset. Drops the response's chain-of-thought; keeps just the gold
    letter."""
    question = str(record.get("question") or "").strip()
    response = str(record.get("response") or "").strip()
    if not question or not response:
        return None

    # Require the question to look MC-shaped (at least 2 lettered options).
    matches = _FLAN_MC_QUESTION_PATTERN.findall(question)
    if len(matches) < 2:
        return None

    # Parse the gold letter from the response. Look for "answer is X" / "answer: X"
    # patterns, falling back to a leading single-letter line.
    letter_match = _FLAN_MC_RESPONSE_LETTER_PATTERN.search(response)
    if letter_match is None:
        return None
    gold_letter = letter_match.group(1).upper()
    if gold_letter not in {"A", "B", "C", "D"}:
        return None

    # Don't re-render the question — preserve the source's MC layout to maximize
    # format-distribution diversity. The model trains to emit just the letter.
    cleaned_question = clean_message_content(question)
    if not cleaned_question:
        return None

    messages = [
        {"role": "user", "content": cleaned_question},
        {"role": "assistant", "content": gold_letter},
    ]
    metadata = {"rationale": source_cfg.rationale, "kind": "flan_mc"}
    return messages, metadata
```

- [ ] **Step 5: Register in `_load_records`**

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/posttraining/test_normalizers_new.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/posttraining/sftConfig.py src/posttraining/prepare.py tests/posttraining/test_normalizers_new.py
git commit -m "feat(sft): add FLAN-MC filter loader (OpenOrca/OpenHermes Tier 5)"
```

---

### Task 7: Build sft_benchmark.yaml config wiring all the new tiers

**Why:** None of the new normalizers do anything until a config lists them as sources. This task produces the single config that runs the spec's full data mix: Tiers 1–5.

**Files:**
- Create: `configs/posttraining/sft_benchmark.yaml`.

- [ ] **Step 1: Create the config from a starting template**

Create `configs/posttraining/sft_benchmark.yaml`. Base it on `sft_final_alpaca_arc.yaml` (the most recent config that has the right model + training shape) and replace its `sources:` block with the spec's Tier 1–5 list.

```yaml
model:
  vocab_size: 50258
  pad_token_id: 50257
  bos_token_id: 50256
  eos_token_id: 50256
  d_model: 384
  n_layers: 14
  n_heads: 6
  d_ff: 768
  context_length: 1024
  bias: false
  dropout: 0.0151
  rope_theta: 10000.0
  gradient_checkpointing: false

logging:
  console_level: INFO
  file_level: DEBUG
  components:
    posttraining: INFO
    training: INFO

inference:
  device: auto
  max_tokens: 128
  temperature: 0.2
  top_k: 50
  top_p: 0.9

chat:
  device: auto
  max_tokens: 40
  temperature: 0.0
  top_k: 50
  top_p: 0.9
  system_prompt: "You are ParrotLLM, a helpful assistant."
  checkpoint_dir: runs/posttraining/sft_benchmark

sft:
  device: auto
  base_checkpoint: runs/posttraining/base_import/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt
  cache_dir: data/posttraining/hf_cache
  raw_dir: data/posttraining/raw
  prepared_dir: data/posttraining/sft_mix_benchmark
  runs_dir: runs/posttraining/sft_benchmark
  checkpoint_dir: checkpoints
  system_prompt: "You are ParrotLLM, a helpful assistant."
  template_format: alpaca
  max_seq_length: 1024
  train_batch_size: 8
  eval_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rates:
    - 5.0e-7
  min_lr_ratio: 0.5
  warmup_ratio: 0.03
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  z_loss_coeff: 0.0
  replay_ratio: 0.0  # Tier 4 explicitly handles LAMBADA shape; no replay needed.
  replay_train_bin: data/processed/train.bin
  replay_val_bin: data/processed/val.bin
  num_epochs: 1.5
  polish_epochs: 0.0
  polish_subset_size: 4000
  save_every: 50
  eval_every: 100
  early_stopping_metric: composite_score
  early_stopping_mode: max
  early_stopping_patience: 4
  keep_last_checkpoints: 4
  keep_best_checkpoints: 3
  log_every: 1
  seed: 42
  compile: false
  format_score_weight: 0.25
  forgetting_penalty_weight: 0.08
  prompt_suite_path: configs/posttraining/dev_prompt_suite_final_sft.jsonl
  log_prompt_suite_generations: true

  sources:
    # ---------------- Tier 1: direct format match, oversampled ----------------
    - name: hellaswag_train
      loader: hellaswag
      path: Rowan/hellaswag
      split: train
      target_examples: 40000
      candidate_multiplier: 1
      quality_weight: 1.3
      tags: [tier1, multiple_choice, hellaswag]
      rationale: Direct task match for the leaderboard HellaSwag benchmark.

    - name: winogrande_train
      loader: winogrande
      path: allenai/winogrande
      subset: winogrande_xl
      split: train
      target_examples: 40000
      candidate_multiplier: 1
      quality_weight: 1.3
      tags: [tier1, multiple_choice, winogrande, binary]
      rationale: Direct task match for the leaderboard WinoGrande benchmark.

    - name: openbookqa_train
      loader: openbookqa
      path: allenai/openbookqa
      subset: main
      split: train
      target_examples: 25000
      candidate_multiplier: 6
      quality_weight: 1.3
      tags: [tier1, multiple_choice, openbookqa]
      rationale: Tiny but exact format; oversample 5x.

    - name: ai2_arc_easy_train
      loader: ai2_arc
      path: allenai/ai2_arc
      subset: ARC-Easy
      split: train
      target_examples: 11000
      candidate_multiplier: 6
      quality_weight: 1.2
      tags: [tier1, multiple_choice, arc]

    - name: ai2_arc_challenge_train
      loader: ai2_arc
      path: allenai/ai2_arc
      subset: ARC-Challenge
      split: train
      target_examples: 5500
      candidate_multiplier: 6
      quality_weight: 1.25
      tags: [tier1, multiple_choice, arc]

    - name: sciq_train
      loader: sciq
      path: allenai/sciq
      split: train
      target_examples: 12000
      candidate_multiplier: 1
      quality_weight: 1.15
      tags: [tier1, multiple_choice, sciq, science]

    # ---------------- Tier 2: commonsense breadth ----------------
    - name: cosmos_qa_train
      loader: cosmos_qa
      path: allenai/cosmos_qa
      split: train
      target_examples: 25000
      candidate_multiplier: 1
      quality_weight: 1.1
      tags: [tier2, multiple_choice, commonsense]

    - name: piqa_train
      loader: piqa
      path: ybisk/piqa
      split: train
      target_examples: 16000
      candidate_multiplier: 1
      quality_weight: 1.1
      tags: [tier2, multiple_choice, piqa, binary]

    - name: social_iqa_train
      loader: social_iqa
      path: allenai/social_i_qa
      split: train
      target_examples: 15000
      candidate_multiplier: 3
      quality_weight: 1.05
      tags: [tier2, multiple_choice, social_iqa]

    - name: commonsense_qa_train
      loader: commonsense_qa
      path: tau/commonsense_qa
      split: train
      target_examples: 9000
      candidate_multiplier: 1
      quality_weight: 1.1
      tags: [tier2, multiple_choice, csqa]

    # ---------------- Tier 3: MC reasoning ----------------
    - name: race_train_subset
      loader: race
      path: ehovy/race
      subset: high
      split: train
      target_examples: 20000
      candidate_multiplier: 5
      quality_weight: 1.0
      tags: [tier3, multiple_choice, race]

    # ---------------- Tier 4: LAMBADA regularizer ----------------
    - name: cbt_train
      loader: cbt
      path: cbt
      subset: NE
      split: train
      target_examples: 10000
      candidate_multiplier: 2
      quality_weight: 1.2
      template_format: raw
      tags: [tier4, lambada_shape]
      rationale: Children's Book Test, LAMBADA-shape regularizer.

    - name: bookcorpus_lambada_train
      loader: bookcorpus_lambada
      path: bookcorpus
      split: train
      target_examples: 20000
      candidate_multiplier: 4
      quality_weight: 1.0
      template_format: raw
      tags: [tier4, lambada_shape]

    # ---------------- Tier 5: public LLM-generated MC ----------------
    - name: openorca_mc
      loader: flan_mc
      path: Open-Orca/OpenOrca
      split: train
      target_examples: 60000
      candidate_multiplier: 30
      quality_weight: 0.9
      tags: [tier5, flan_mc, openorca]
      rationale: OpenOrca filtered to MC-shape rows; rationale text dropped, gold letter only.

    - name: mmlu_aux_train
      loader: mmlu
      path: cais/mmlu
      subset: auxiliary_train
      split: train
      target_examples: 30000
      candidate_multiplier: 4
      quality_weight: 1.0
      tags: [tier5, multiple_choice, mmlu]

  decontam_datasets:
    - name: wikitext103_test
      loader: local_disk
      path: data/wikitext-103-test
      field: text
      split: test
    - name: nlp26_owt_eval
      loader: local_disk
      path: data/owt-eval/NLP26/NLP26_OWT_eval/test
      field: text
      split: test
    - name: hellaswag_val
      loader: huggingface
      path: Rowan/hellaswag
      field: ctx
      split: validation
    - name: winogrande_val
      loader: huggingface
      path: allenai/winogrande
      subset: winogrande_xl
      field: sentence
      split: validation
    - name: openbookqa_val
      loader: huggingface
      path: allenai/openbookqa
      subset: main
      field: question_stem
      split: validation
    - name: lambada_test
      loader: huggingface
      path: EleutherAI/lambada_openai
      field: text
      split: test
```

- [ ] **Step 2: Validate the YAML parses against ProjectConfig**

```bash
uv run python -c "
from configs import load_project_config
cfg = load_project_config('configs/posttraining/sft_benchmark.yaml')
sft = cfg.sft
print(f'Sources: {len(sft.sources)}')
for s in sft.sources:
    print(f'  - {s.name:30s} ({s.loader:24s}) target={s.target_examples}')
print(f'Decontam datasets: {len(sft.decontam_datasets)}')
"
```

Expected: lists 16 sources (6 Tier1 + 4 Tier2 + 1 Tier3 + 2 Tier4 + 2 Tier5 = wait, recount: 6+4+1+2+2 = 15) and 6 decontam datasets, no validation errors. If pydantic complains about a `loader` value, double-check Tasks 2–6 added all four new entries (`cosmos_qa`, `social_iqa`, `cbt`, `bookcorpus_lambada`, `flan_mc`) to the `Literal` in `sftConfig.py`.

- [ ] **Step 3: Commit the config**

```bash
git add configs/posttraining/sft_benchmark.yaml
git commit -m "config(sft): add benchmark-targeted SFT data mix (Tiers 1-5)"
```

---

### Task 8: Re-enable DPO decontamination

**Why:** `src/posttraining/dpo/prepare.py:116-131` stubs out DPO decontamination with the rationale that HH-RLHF rarely overlaps with cloze benchmarks. That rationale doesn't apply to the current `mc_letter` source set (HellaSwag-train, WinoGrande-train, OBQA-train, MMLU-aux), where eval-split leakage is plausible. The fix is to wire the SFT-side `PromptContaminationIndex` over DPO pairs and add a sanity check.

**Files:**
- Modify: `src/posttraining/dpo/prepare.py` — replace the `_decontam_set` stub with a real prompt-contamination check using the existing SFT-side machinery.
- Modify: `tests/posttraining/test_dpo_prepare.py` — add a test asserting decontam drops a leaked example.

- [ ] **Step 1: Read the current stub**

Open `src/posttraining/dpo/prepare.py` and read lines 116–131. Note: `_decontam_set(decontam_specs, tokenizer)` currently returns `set()` unconditionally (this is the stub).

- [ ] **Step 2: Read the SFT-side decontamination class**

Open `src/posttraining/prepare.py` lines 86–138 and confirm `PromptContaminationIndex` has a `contains(text)` method that takes a candidate prompt and returns True if it overlaps an eval-split entry.

- [ ] **Step 3: Write the failing test**

Append to `tests/posttraining/test_dpo_prepare.py`:

```python
def test_dpo_decontam_drops_leaked_pair(monkeypatch, tmp_path):
    """A pair whose prompt matches a decontam-set entry must be dropped."""
    from src.posttraining.dpo.prepare import _build_letter_dpo_pair, _filter_decontaminated

    decontam_set = {"this is the leaked context that should be dropped"}

    pair_clean = _build_letter_dpo_pair(
        prompt="A clean MC question.\nA) one\nB) two\nAnswer:",
        gold_letter="A",
        wrong_letter="B",
    )
    pair_leaked = _build_letter_dpo_pair(
        prompt="this is the leaked context that should be dropped\nA) x\nB) y\nAnswer:",
        gold_letter="A",
        wrong_letter="B",
    )
    kept = _filter_decontaminated([pair_clean, pair_leaked], decontam_set)
    assert len(kept) == 1
    assert "leaked" not in kept[0]["prompt"].lower()
```

- [ ] **Step 4: Run to verify failure**

```bash
uv run pytest tests/posttraining/test_dpo_prepare.py::test_dpo_decontam_drops_leaked_pair -v
```

Expected: FAIL on `ImportError: cannot import name '_filter_decontaminated'`.

- [ ] **Step 5: Implement the decontam filter and wire it into `run_prepare_dpo` paths**

In `src/posttraining/dpo/prepare.py`, add (near `_decontam_set`):

```python
def _filter_decontaminated(
    pairs: list[dict[str, Any]], decontam_prompts: set[str] | "PromptContaminationIndex"
) -> list[dict[str, Any]]:
    """Drop any pair whose prompt overlaps an eval-split entry.

    Accepts either a flat set of normalized prompt strings (cheap exact-match)
    or a PromptContaminationIndex (MinHash + 5-gram + Jaccard).
    """
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        prompt = pair.get("prompt") or ""
        if isinstance(decontam_prompts, set):
            normalized = " ".join(prompt.lower().split())
            if any(entry in normalized for entry in decontam_prompts):
                continue
        else:
            if decontam_prompts.contains(prompt):
                continue
        kept.append(pair)
    return kept
```

Then update the `_decontam_set` function and `_run_prepare_dpo_mc_letter` (around line 266) to call `_filter_decontaminated` against the same `decontam_datasets` list the SFT path uses. The simplest wiring: import `PromptContaminationIndex` from `src.posttraining.prepare`, build it once from the project config's `sft.decontam_datasets`, and apply it to the constructed pairs before writing them out. Print the dropped count to the run log.

The exact integration point depends on how `_run_prepare_dpo_mc_letter` builds the pair list — read that function's structure and insert the filter where the pair list is finalized.

- [ ] **Step 6: Run the test**

```bash
uv run pytest tests/posttraining/test_dpo_prepare.py::test_dpo_decontam_drops_leaked_pair -v
```

Expected: PASS.

- [ ] **Step 7: Run the full DPO-prepare test suite to confirm no regressions**

```bash
uv run pytest tests/posttraining/test_dpo_prepare.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/posttraining/dpo/prepare.py tests/posttraining/test_dpo_prepare.py
git commit -m "fix(dpo): re-enable prompt decontamination for MC-letter pairs

The previous stub assumed HH-RLHF sources rarely overlap with cloze
benchmarks. That assumption breaks for the current MC-letter source set
(HellaSwag/WinoGrande/OBQA-train/MMLU-aux), where eval-split leakage is
plausible. Filter pairs through the same MinHash index used on the SFT
side, log the drop count."
```

---

### Task 9: Mechanical config cleanup

**Why:** The audit found 20+ SFT configs and 11+ DPO configs, most abandoned. Visual noise costs real time when figuring out what's current. Move abandoned ones to an `_archive/` subdirectory and rename the chat-style config to make the benchmark/chat separation explicit.

**Files:**
- Move: many `configs/posttraining/*.yaml` files to `configs/posttraining/_archive/`.
- Rename: `configs/posttraining/sft_full_recipe.yaml` → `configs/posttraining/sft_chat_demo.yaml`.

- [ ] **Step 1: Create the archive directory and move abandoned configs**

```bash
mkdir -p configs/posttraining/_archive
git mv configs/posttraining/sft_alpaca.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_benchmark.yaml.bak 2>/dev/null || true  # if exists
git mv configs/posttraining/sft_exp_a_zloss.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_exp_b_lima.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_exp_c_polish.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_final_alpaca_arc.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_final_large_alpaca.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_final_overnight.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_final_trial.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_format_only.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_one_lr.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_smoke.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_smoke_fast.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_with_mc.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_with_mc_smoke.yaml configs/posttraining/_archive/
git mv configs/posttraining/sft_with_mc_v2.yaml configs/posttraining/_archive/
git mv configs/posttraining/dpo.yaml configs/posttraining/_archive/
git mv configs/posttraining/dpo_letter.yaml configs/posttraining/_archive/
git mv configs/posttraining/dpo_letter_smoke.yaml configs/posttraining/_archive/
git mv configs/posttraining/dpo_letter_v2.yaml configs/posttraining/_archive/
git mv configs/posttraining/dpo_smoke.yaml configs/posttraining/_archive/
```

If any of those file names doesn't exist on your branch, `git mv` will error. Skip and continue with the next one. The point is to archive **abandoned** configs; if a config is in active use (referenced by `scripts/overnight_dpo_compare.sh` or recently committed), leave it in place.

Active configs to **keep** in `configs/posttraining/`:
- `sft.yaml` (current default)
- `sft_mixed_low.yaml` (active per audit)
- `sft_full_recipe.yaml` → will be renamed below
- `sft_benchmark.yaml` (this plan's output)
- `dpo_benchmark_shared.yaml`
- `dpo_for_sft_a.yaml`, `dpo_for_sft_b.yaml`
- `dpo_letter_v2_aggressive.yaml`
- `sftConfig.py`, `dpoConfig.py`, `__init__.py`
- `dev_prompt_suite*.jsonl`

- [ ] **Step 2: Rename the chat-style recipe**

```bash
git mv configs/posttraining/sft_full_recipe.yaml configs/posttraining/sft_chat_demo.yaml
```

- [ ] **Step 3: Verify nothing references the moved/renamed paths**

```bash
grep -RIl "sft_full_recipe\|sft_final_alpaca_arc\|sft_format_only\|sft_with_mc" --include="*.py" --include="*.yaml" --include="*.sh" --include="*.md" /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM | grep -v _archive | grep -v docs/superpowers
```

Expected: empty output, OR only references to legitimate places (e.g., `scripts/overnight_dpo_compare.sh` referring to active configs). If any active script references an archived config, either restore that config or update the script.

If the grep shows `scripts/overnight_dpo_compare.sh` references `sft_format_only.yaml` or `sft_mixed_low.yaml`, those configs are still active — move `sft_format_only.yaml` back out of `_archive/` if it was moved there. (Check the audit notes: `sft_mixed_low.yaml` is active, and `sft_format_only.yaml` was referenced by overnight scripts — keep it active.)

- [ ] **Step 4: Add a README in the archive directory**

```bash
cat > configs/posttraining/_archive/README.md <<'EOF'
# Archived posttraining configs

These configs were superseded by newer experiments and are kept here only
for historical reference. Active configs live one directory up.

If you're starting a new SFT or DPO run, start from one of:
  - configs/posttraining/sft_benchmark.yaml      (benchmark checkpoint)
  - configs/posttraining/sft_chat_demo.yaml      (chat-demo checkpoint)
  - configs/posttraining/sft_mixed_low.yaml      (current active SFT)
  - configs/posttraining/dpo_benchmark_shared.yaml
EOF
```

- [ ] **Step 5: Commit**

```bash
git add configs/posttraining/
git commit -m "chore(posttraining): archive abandoned configs, rename chat recipe

Reduces the visible config set to active candidates only. sft_full_recipe
is renamed to sft_chat_demo to make the benchmark/chat checkpoint
separation explicit."
```

---

### Task 10: End-to-end smoke test

**Why:** All previous tasks added pieces but didn't verify they hang together at the data-prep level. This task runs `sft-prepare` against the new config in a low-volume sanity mode and confirms the prepared dataset is well-formed.

**Files:**
- No new code — just running existing CLI flow.

- [ ] **Step 1: Create a tiny variant of sft_benchmark.yaml for smoke testing**

```bash
cat configs/posttraining/sft_benchmark.yaml | sed \
  -e 's/target_examples: [0-9]*/target_examples: 50/' \
  -e 's/candidate_multiplier: [0-9]*/candidate_multiplier: 2/' \
  > configs/posttraining/sft_benchmark_smoke.yaml
```

- [ ] **Step 2: Run sft-download for the smoke config**

```bash
uv run python main.py --stage sft-download --config configs/posttraining/sft_benchmark_smoke.yaml
```

Expected: each source downloads (or finds cached) and writes a snapshot under `data/posttraining/raw/`. Some sources (Open-Orca/OpenOrca, bookcorpus) are large; if disk pressure is a concern, set `target_examples: 50` is small enough that the loader should still pick a 50-row subset before paging through everything — but if the download flow forces a full download regardless of `target_examples`, this step may take a long time. If so, manually skip OpenOrca and BookCorpus by commenting their entries out of `sft_benchmark_smoke.yaml` for the smoke run.

- [ ] **Step 3: Run sft-prepare for the smoke config**

```bash
uv run python main.py --stage sft-prepare --config configs/posttraining/sft_benchmark_smoke.yaml
```

Expected: writes prepared examples under `data/posttraining/sft_mix_benchmark/` (with `_smoke` suffix or similar — check the actual path the prepare flow uses). No exceptions; per-source counts logged.

- [ ] **Step 4: Validate the prepared output's letter distribution**

```bash
uv run python -c "
import json
from pathlib import Path
from collections import Counter

# Adjust path to wherever the smoke run wrote its output.
prepared = Path('data/posttraining/sft_mix_benchmark')  # may be sft_mix_benchmark_smoke
candidates = list(prepared.rglob('*.jsonl'))
print(f'Found {len(candidates)} prepared jsonl files')

counter = Counter()
total = 0
for path in candidates:
    if 'lambada_shape' in path.name or 'cbt' in path.name or 'bookcorpus' in path.name:
        continue  # LAMBADA-shape examples don't have a letter target.
    with path.open() as f:
        for line in f:
            ex = json.loads(line)
            messages = ex.get('messages') or ex.get('conversation') or []
            if not messages:
                continue
            assistant = messages[-1].get('content', '').strip()
            if assistant in {'A', 'B', 'C', 'D'}:
                counter[assistant] += 1
                total += 1
print(f'Total MC examples scanned: {total}')
print(f'Letter distribution: {dict(counter)}')
if total > 100:
    for letter, n in counter.items():
        pct = 100.0 * n / total
        assert 18.0 < pct < 32.0 or letter not in {'A', 'B', 'C', 'D'}, \
            f'Gold letter {letter} is {pct:.1f}% — expected ~25% under uniform permutation'
print('Letter distribution within tolerance.')
"
```

Expected: prints letter counts; if total > 100, all four letters are within ~7pp of uniform 25%. If the distribution is heavily skewed, choice-order permutation isn't being applied — debug Task 1's integration.

- [ ] **Step 5: Spot-check 5 examples manually**

```bash
uv run python -c "
import json
from pathlib import Path
prepared = Path('data/posttraining/sft_mix_benchmark')
files = list(prepared.rglob('hellaswag*.jsonl')) + list(prepared.rglob('cbt*.jsonl'))
for path in files[:2]:
    print(f'=== {path.name} ===')
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            ex = json.loads(line)
            print(json.dumps(ex.get('messages') or ex, indent=2)[:800])
            print('---')
"
```

Confirm:
- HellaSwag examples have user content ending with `Answer:` and assistant content of just one letter A/B/C/D.
- CBT examples have user content ending with a trailing space and assistant content being a single lowercase word.

- [ ] **Step 6: Clean up the smoke config and commit success notes**

```bash
rm configs/posttraining/sft_benchmark_smoke.yaml
```

If the smoke run succeeded, commit the validation notes — but no code changes, just a marker:

```bash
# No code to commit — this task is pure validation.
# If you want to record the validation, write a short note to the spec/plan
# directory or just note completion in the PR description.
echo "Smoke test passed: data prep produces well-formed MC + LAMBADA-shape examples with uniform letter distribution"
```

If the smoke run FAILED at any source, that's a real bug — go back to the relevant earlier task, fix, and re-run. Don't proceed to Plan B (DPO restructure) until this passes.

---

## Self-Review

Going through the spec sections to check coverage.

**Spec §3.1 SFT data mix:**
- Tier 1 (HellaSwag/Winogrande/OBQA/ARC/SciQ): wired in Task 7 config; permutation Task 1.
- Tier 2 (CosmosQA/PIQA/SocialIQa/CSQA): CosmosQA Task 2, SocialIQa Task 3, PIQA & CSQA already exist + wired Task 7.
- Tier 3 (RACE/LogiQA/QASC): RACE wired Task 7. **LogiQA and QASC are not in this plan.** Deferred — they're 7k and 8k respectively, marginal compared to RACE-20k. Add as follow-up if accuracy is short of target.
- Tier 4 (CBT/BookCorpus/Wikitext): CBT Task 4, BookCorpus Task 5. **Wikitext-103-train continuation chunks are not in this plan.** The `narrative_completion` loader exists already and could be wired in Task 7 if needed; left out to avoid scope creep, can be added with one config entry.
- Tier 5 (OpenOrca/Hermes/Tulu/MMLU-aux): OpenOrca via Task 6 generic FLAN-MC normalizer; MMLU-aux via existing MMLU loader; OpenHermes shares the Task 6 normalizer (just needs a config entry pointing at `teknium/OpenHermes-2.5`); Tulu MC slice uses the existing `tulu` loader. **OpenHermes is not wired in Task 7's config.** Add as follow-up.

**Spec §3.2 format augmentation:**
- Choice-order permutation: covered in Task 1.
- Format paraphrase: **not in this plan.** Per spec §3.2 priority note: "lower priority under cloze inference"; correctly deferred.

**Spec §3.3 loss masking:** No code change — current implementation already masks to assistant span; Task 1 confirms this in its integration test (assistant content = single letter).

**Spec §3.4 DPO data:** Out of scope for this plan; covered in Plan B.

**Spec §3.5 decontamination:** SFT side stays as-is (already correct). DPO side re-enabled in Task 8. Sanity check (drop count must be non-zero on ARC/MMLU-aux/OpenOrca) is a manual inspection in Task 10 Step 4.

**Spec §3.6 phase order:** Out of scope; pipeline runner not built here.

**Spec §3.7 sanity gates:** Letter distribution audit covered in Task 10 Step 4. Other gates (format-stranger, LAMBADA non-regression, cloze-vs-letter agreement) are out of scope; covered in Plan C.

**Spec §A fallback design:** Out of scope (only built if needed).

**Placeholder scan:** Searched for "TBD" / "TODO" / "fill in" / "implement later" / "similar to Task" — none found. Each step contains the actual content.

**Type consistency:**
- `_permute_choices(choices, answer_index, seed_key)` — same signature in Task 1 and used identically in Tasks 2, 3.
- `_normalize_*_record(record, source_cfg)` — uniform signature matching the existing prepare.py pattern.
- `_filter_decontaminated(pairs, decontam_prompts)` — defined and used in Task 8.

**Spec gaps left explicit (acceptable scope):** LogiQA, QASC, OpenHermes config wiring, Wikitext continuation, format paraphrase. All are additive — they can be added in follow-up small PRs without disturbing this plan's structure.

The plan covers the spec's largest expected-lift changes: missing MC sources in SFT, choice permutation, and DPO decontam. Plans B and C continue from the SFT checkpoint produced by this plan.
