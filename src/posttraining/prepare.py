"""Dataset preparation pipeline for SFT posttraining."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from datasets import load_dataset, load_from_disk

from configs import ProjectConfig, SFTDecontamConfig, SFTSourceConfig
from .templates import TokenizedConversation, clean_message_content, trim_messages_to_token_limit
from src.utils import build_tokenizer


log = logging.getLogger("parrotllm.posttraining")

_LANG_MODEL_PATH = Path("data/lid.176.ftz")
_WORD_RE = re.compile(r"\w+")
_PLACEHOLDER_RE = re.compile(r"(?:\[\s*(?:insert|todo|placeholder)[^\]]*\]|<\s*(?:insert|todo|placeholder)[^>]*>)", re.IGNORECASE)
_REDaction_RE = re.compile(r"\[(?:redacted|removed)\]", re.IGNORECASE)
_CHAIN_OF_THOUGHT_RE = re.compile(
    r"(?:let'?s think step by step|chain of thought|step 1[:.]|first,|second,|therefore,)",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# Mirrors `normalize_lambada` in external/PikoGPT_Leaderboard/leaderboard/run_benchmarks.py
# so the train-side cloze target uses the same stripping rule the eval-side
# applies to the model's prediction. Reused by the BookCorpus normalizer.
_LAMBADA_STRIP_CHARS = " \t\r\n\"'“”‘’.,;:!?()[]{}"


@dataclass(slots=True)
class PreparedExample:
    source: str
    tags: list[str]
    quality_score: float
    prompt_hash: str
    prompt_text: str
    full_text_hash: str
    tokens: list[int]
    loss_mask: list[int]
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


class OptionalLanguageFilter:
    def __init__(self, target_language: str = "en", model_path: Path = _LANG_MODEL_PATH):
        self.target_language = target_language.lower()
        self._model = None
        self.available = False
        if model_path.exists():
            try:
                import fasttext

                self._model = fasttext.load_model(str(model_path))
                self.available = True
            except Exception as exc:  # pragma: no cover - depends on local setup
                log.warning("Failed to load language detector from %s: %s", model_path, exc)

    def matches(self, text: str, expected: str | None) -> bool:
        if expected is None:
            return True
        expected = expected.lower()
        if not self.available or self._model is None:
            return True
        snippet = text.replace("\n", " ").strip()
        if not snippet:
            return False
        try:
            label, confidence = self._model.predict(snippet[:1000], k=1)
        except Exception as exc:  # pragma: no cover - depends on fasttext/numpy runtime
            log.warning("Language detector failed during SFT prepare; disabling filter: %s", exc)
            self.available = False
            self._model = None
            return True
        lang = label[0].replace("__label__", "").lower()
        return lang == expected and float(confidence[0]) >= 0.70


class PromptContaminationIndex:
    def __init__(self, *, threshold: float = 0.8, num_perm: int = 16, bands: int = 4):
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows = max(1, num_perm // bands)
        self._prime = (1 << 61) - 1
        self._a = [1_000_003 + i * 1_009 for i in range(num_perm)]
        self._b = [7_919 + i * 37 for i in range(num_perm)]
        self._bucket_to_indices: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        self._shingles: list[set[int]] = []
        self._exact_hashes: set[str] = set()

    def add(self, text: str) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return
        self._exact_hashes.add(_stable_hash(normalized))
        shingles = _shingle_hashes(normalized)
        index = len(self._shingles)
        self._shingles.append(shingles)
        signature = self._signature(shingles)
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            band = tuple(signature[start : start + self.rows])
            self._bucket_to_indices[(band_idx, band)].append(index)

    def contains(self, text: str) -> bool:
        normalized = _normalize_text(text)
        if not normalized:
            return False
        if _stable_hash(normalized) in self._exact_hashes:
            return True
        shingles = _shingle_hashes(normalized)
        signature = self._signature(shingles)
        candidates: set[int] = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            band = tuple(signature[start : start + self.rows])
            candidates.update(self._bucket_to_indices.get((band_idx, band), ()))
        for idx in candidates:
            if _jaccard(shingles, self._shingles[idx]) >= self.threshold:
                return True
        return False

    def _signature(self, shingles: set[int]) -> list[int]:
        if not shingles:
            return [self._prime - 1] * self.num_perm
        signature: list[int] = []
        for a, b in zip(self._a, self._b):
            signature.append(min((a * value + b) % self._prime for value in shingles))
        return signature


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    cleaned = clean_message_content(text).lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _shingle_hashes(text: str, size: int = 5) -> set[int]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < size:
        return set()
    values: set[int] = set()
    for idx in range(len(words) - size + 1):
        shingle = " ".join(words[idx : idx + size]).encode("utf-8")
        digest = hashlib.blake2b(shingle, digest_size=8).digest()
        values.add(int.from_bytes(digest, "big"))
    return values


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for message in messages:
        content = clean_message_content(message.get("content"))
        content = _PLACEHOLDER_RE.sub("", content)
        content = _REDaction_RE.sub("", content)
        if not content:
            continue
        sanitized.append({"role": message["role"], "content": content})
    return sanitized


def _normalize_source_key(value: str | None) -> str:
    if value is None:
        return ""
    return _NON_ALNUM_RE.sub("", str(value).lower())


def _source_prompt_text(messages: list[dict[str, str]]) -> str:
    parts = [m["content"] for m in messages if m["role"] != "assistant"]
    return "\n".join(parts).strip()


def _quality_score(messages: list[dict[str, str]], *, source_cfg: SFTSourceConfig) -> float:
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    answer_words = sum(len(text.split()) for text in assistants)
    score = float(source_cfg.quality_weight)
    if 6 <= answer_words <= 180:
        score += 0.05
    if len(messages) >= 4:
        score += 0.03
    if any(tag in {"json", "extraction"} for tag in source_cfg.tags):
        score += 0.02
    if any(_CHAIN_OF_THOUGHT_RE.search(text) for text in assistants):
        score -= 0.15
    return score


def _looks_like_chain_of_thought(text: str) -> bool:
    return bool(_CHAIN_OF_THOUGHT_RE.search(text))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "toxic", "harmful", "unsafe"}
    return False


def _iter_local_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _snapshot_component(value: str | None) -> str:
    normalized = _NON_ALNUM_RE.sub("_", str(value or "").lower()).strip("_")
    return normalized or "default"


def _hf_snapshot_stem(path: str, subset: str | None, split: str) -> str:
    return "__".join(
        [
            _snapshot_component(path),
            _snapshot_component(subset),
            _snapshot_component(split),
        ]
    )


def get_source_snapshot_path(raw_dir: Path, source_cfg: SFTSourceConfig) -> Path | None:
    if source_cfg.loader == "local_jsonl":
        return None
    return raw_dir / "sources" / _hf_snapshot_stem(
        source_cfg.path,
        source_cfg.subset,
        source_cfg.split,
    )


def get_decontam_snapshot_path(raw_dir: Path, cfg: SFTDecontamConfig) -> Path | None:
    if cfg.loader != "huggingface":
        return None
    return raw_dir / "decontam" / _hf_snapshot_stem(
        cfg.path,
        cfg.subset,
        cfg.split,
    )


def _load_records(
    source_cfg: SFTSourceConfig,
    *,
    snapshot_path: Path | None = None,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> Iterable[Mapping[str, Any]]:
    if source_cfg.loader == "local_jsonl":
        path = Path(source_cfg.path)
        if not path.exists():
            log.warning("Custom SFT file missing: %s", path)
            return []
        return list(_iter_local_jsonl(path))
    if snapshot_path is not None:
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"SFT dataset snapshot missing for source '{source_cfg.name}' at {snapshot_path}. "
                "Run `--stage sft-download` first."
            )
        return load_from_disk(str(snapshot_path))
    load_kwargs: dict[str, Any] = {"split": source_cfg.split}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    if hf_token is not None:
        load_kwargs["token"] = hf_token
    return load_dataset(
        source_cfg.path,
        source_cfg.subset,
        **load_kwargs,
    )


def _normalize_local_jsonl_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    if "messages" in record:
        messages = [
            {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
            for message in record.get("messages", [])
            if isinstance(message, Mapping)
        ]
    else:
        prompt = str(record.get("prompt", "")).strip()
        completion = str(record.get("completion", "") or record.get("response", "")).strip()
        if not prompt or not completion:
            return None
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    metadata = {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }
    return messages, metadata


def _normalize_alpaca_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    instruction = str(record.get("instruction") or record.get("prompt") or "").strip()
    input_text = str(record.get("input") or record.get("context") or "").strip()
    response = str(
        record.get("output")
        or record.get("response")
        or record.get("completion")
        or ""
    ).strip()
    if not instruction or not response:
        return None

    prompt = instruction
    if input_text:
        prompt = f"{instruction}\n\nInput:\n{input_text}"

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    metadata = {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }
    return messages, metadata


def _normalize_ai2_arc_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    question = str(record.get("question") or "").strip()
    raw_choices = record.get("choices")
    answer_key = str(record.get("answerKey") or record.get("answer_key") or "").strip()
    if not question or not isinstance(raw_choices, Mapping) or not answer_key:
        return None

    choice_texts = raw_choices.get("text")
    if not isinstance(choice_texts, list) or len(choice_texts) != 4:
        return None

    cleaned_choices = [clean_message_content(text) for text in choice_texts]
    if any(not text for text in cleaned_choices):
        return None

    raw_labels = raw_choices.get("label")
    answer_index: int | None = None
    if isinstance(raw_labels, list):
        normalized_answer = answer_key.strip().upper()
        for idx, label in enumerate(raw_labels):
            label_text = str(label).strip().upper()
            if label_text == normalized_answer:
                answer_index = idx
                break

    if answer_index is None:
        if answer_key.isdigit():
            numeric_index = int(answer_key) - 1
            if 0 <= numeric_index < len(cleaned_choices):
                answer_index = numeric_index
        else:
            letter_index = ord(answer_key.upper()[:1]) - ord("A")
            if 0 <= letter_index < len(cleaned_choices):
                answer_index = letter_index

    if answer_index is None:
        return None

    seed_key = str(record.get("id") or question)
    cleaned_choices, answer_index = _permute_choices(
        cleaned_choices, answer_index, seed_key
    )
    prompt = _format_mc_prompt(question, cleaned_choices, prefix="Question")
    messages = _build_mc_messages(prompt, "ABCD"[answer_index])
    metadata = {
        "record_id": record.get("id"),
        "arc_subset": source_cfg.subset,
        "original_answer_key": answer_key,
        "rationale": source_cfg.rationale,
    }
    return messages, metadata


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


def _format_mc_prompt(question: str, choices: list[str], prefix: str | None = "Question") -> str:
    """Format an MC prompt to match the PikoGPT leaderboard's eval template.

    Eval prompts (from external/PikoGPT_Leaderboard/.../validation.jsonl):
      HellaSwag/WinoGrande -> "Context: <text>\\nA) ...\\nAnswer:"
      OpenBookQA           -> "Question: <stem>\\nA) ...\\nAnswer:"
    """
    labels = ["A", "B", "C", "D", "E"]
    options = "\n".join(f"{labels[i]}) {choices[i]}" for i in range(len(choices)))
    head = f"{prefix}: {question.strip()}" if prefix else question.strip()
    return f"{head}\n{options}\nAnswer:"


def _build_mc_messages(prompt_text: str, answer_letter: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": answer_letter},
    ]


def _normalize_sciq_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    question = str(record.get("question") or "").strip()
    correct = str(record.get("correct_answer") or "").strip()
    distractors = [str(record.get(k) or "").strip() for k in ("distractor1", "distractor2", "distractor3")]
    if not question or not correct or any(not d for d in distractors):
        return None
    seed = int(hashlib.sha1(question.encode("utf-8")).hexdigest(), 16) % (2**31)
    import random

    rng = random.Random(seed)
    options = [correct, *distractors]
    rng.shuffle(options)
    answer_index = options.index(correct)
    prompt = _format_mc_prompt(question, options)
    answer_letter = "ABCD"[answer_index]
    return _build_mc_messages(prompt, answer_letter), {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }


def _normalize_commonsense_qa_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    question = str(record.get("question") or "").strip()
    raw_choices = record.get("choices")
    answer_key = str(record.get("answerKey") or "").strip().upper()
    if not question or not isinstance(raw_choices, Mapping) or not answer_key:
        return None
    labels = raw_choices.get("label") or []
    texts = raw_choices.get("text") or []
    if not isinstance(texts, list) or len(texts) != 5 or not isinstance(labels, list) or len(labels) != 5:
        return None
    cleaned = [str(t).strip() for t in texts]
    if any(not c for c in cleaned):
        return None
    try:
        answer_index = [str(l).strip().upper() for l in labels].index(answer_key)
    except ValueError:
        return None
    seed_key = str(record.get("id") or question)
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)
    prompt = _format_mc_prompt(question, cleaned)
    return _build_mc_messages(prompt, "ABCDE"[answer_index]), {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }


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


def _normalize_race_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    article = str(record.get("article") or "").strip()
    question = str(record.get("question") or "").strip()
    options = record.get("options") or []
    answer = str(record.get("answer") or "").strip().upper()
    if not article or not question or not isinstance(options, list) or len(options) != 4 or not answer:
        return None
    cleaned = [str(o).strip() for o in options]
    if any(not c for c in cleaned) or answer not in "ABCD":
        return None
    answer_index = "ABCD".index(answer)
    prompt_text = f"Passage: {article}\n\nQuestion: {question}"
    seed_key = str(record.get("example_id") or record.get("id") or prompt_text[:64])
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)
    prompt = _format_mc_prompt(prompt_text, cleaned, prefix=None)
    return _build_mc_messages(prompt, "ABCD"[answer_index]), {
        "record_id": record.get("example_id") or record.get("id"),
        "rationale": source_cfg.rationale,
    }


def _normalize_mmlu_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    # `cais/mmlu` config `auxiliary_train` wraps each row as {"train": {...}}.
    # Other splits/configs are flat. Unwrap if needed.
    inner = record.get("train")
    if isinstance(inner, Mapping):
        record = inner
    question = str(record.get("question") or "").strip()
    choices = record.get("choices") or []
    answer = record.get("answer")
    if not question or not isinstance(choices, list) or len(choices) != 4 or answer is None:
        return None
    cleaned = [str(c).strip() for c in choices]
    if any(not c for c in cleaned):
        return None
    try:
        answer_index = int(answer)
    except (TypeError, ValueError):
        return None
    if not (0 <= answer_index < 4):
        return None
    seed_key = str(record.get("question") or "")[:128]
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)
    prompt = _format_mc_prompt(question, cleaned)
    return _build_mc_messages(prompt, "ABCD"[answer_index]), {
        "record_id": record.get("id"),
        "subject": record.get("subject"),
        "rationale": source_cfg.rationale,
    }


def _normalize_boolq_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    question = str(record.get("question") or "").strip()
    passage = str(record.get("passage") or "").strip()
    answer = record.get("answer")
    if not question or not passage or answer is None:
        return None
    answer_letter = "A" if bool(answer) else "B"
    prompt_text = f"Passage: {passage}\n\nQuestion: {question}"
    prompt = _format_mc_prompt(prompt_text, ["Yes", "No"], prefix=None)
    return _build_mc_messages(prompt, answer_letter), {
        "record_id": record.get("id") or record.get("idx"),
        "rationale": source_cfg.rationale,
    }


def _normalize_piqa_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    goal = str(record.get("goal") or "").strip()
    sol1 = str(record.get("sol1") or "").strip()
    sol2 = str(record.get("sol2") or "").strip()
    label = record.get("label")
    if not goal or not sol1 or not sol2 or label is None:
        return None
    try:
        answer_index = int(label)
    except (TypeError, ValueError):
        return None
    if answer_index not in (0, 1):
        return None
    options = [sol1, sol2]
    seed_key = str(record.get("goal") or "")
    options, answer_index = _permute_choices(options, answer_index, seed_key)
    answer_letter = "AB"[answer_index]
    prompt = _format_mc_prompt(goal, options)
    return _build_mc_messages(prompt, answer_letter), {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }


def _normalize_wsc273_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    text = str(record.get("text") or "").strip()
    pronoun = str(record.get("pronoun") or "").strip()
    options = record.get("options") or []
    label = record.get("label")
    if not text or not pronoun or not isinstance(options, list) or len(options) != 2 or label is None:
        return None
    cleaned = [str(o).strip() for o in options]
    if any(not c for c in cleaned):
        return None
    try:
        answer_index = int(label)
    except (TypeError, ValueError):
        return None
    if answer_index not in (0, 1):
        return None
    question = f'{text}\nIn the above sentence, the pronoun "{pronoun}" refers to:'
    prompt = _format_mc_prompt(question, cleaned, prefix="Context")
    answer_letter = "AB"[answer_index]
    return _build_mc_messages(prompt, answer_letter), {
        "record_id": record.get("source") or "wsc273",
        "rationale": source_cfg.rationale,
    }


def _normalize_hellaswag_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    ctx = str(record.get("ctx") or record.get("ctx_a") or "").strip()
    endings = record.get("endings") or []
    label = record.get("label")
    if not ctx or not isinstance(endings, list) or len(endings) != 4 or label in (None, ""):
        return None
    cleaned = [str(e).strip() for e in endings]
    if any(not c for c in cleaned):
        return None
    try:
        answer_index = int(label)
    except (TypeError, ValueError):
        return None
    if answer_index not in (0, 1, 2, 3):
        return None
    seed_key = str(record.get("ind") or record.get("source_id") or ctx[:64])
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)
    prompt = _format_mc_prompt(ctx, cleaned, prefix="Context")
    answer_letter = "ABCD"[answer_index]
    return _build_mc_messages(prompt, answer_letter), {
        "record_id": record.get("ind") or record.get("source_id"),
        "rationale": source_cfg.rationale,
    }


def _normalize_narrative_completion_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Build a (passage_minus_last_word -> last_word) pair for LAMBADA-shape SFT.

    Eval (LAMBADA test) prompts are raw narrative passages ending mid-sentence;
    the model must produce the missing word. We mirror that shape: instruction
    is the passage minus its final word, response is the final word. With the
    Alpaca wrapper applied at training and inference (main.py:alpaca_wrap),
    the surface form matches eval after the wrapper is added.

    Picks a single trailing-word split from the END of the record text. Rejects
    records that are too short, end on punctuation only, or whose last word is
    a single character. Source `text` field is configurable via record key.
    """
    text = str(record.get("text") or record.get("story") or record.get("content") or "").strip()
    if len(text) < 200:
        return None
    # Strip trailing whitespace/quotes; keep terminal punctuation for now.
    body = text.rstrip()
    # Split on the last whitespace before any trailing punctuation cluster.
    # We want: "<prefix> <word><optional terminal punctuation>"
    # Find the last word boundary.
    import re

    m = re.search(r"\s+([\w'-]+)\s*[\.\?!,]?\s*$", body)
    if not m:
        return None
    last_word = m.group(1)
    if len(last_word) < 2:
        return None
    prefix = body[: m.start()].rstrip()
    if len(prefix) < 100:
        return None
    # Truncate prefix to a manageable length (chars; tokenizer-level cap is
    # enforced later by trim_messages_to_token_limit).
    if len(prefix) > 1500:
        prefix = prefix[-1500:]
        # Snap to the nearest leading sentence boundary so we don't start
        # mid-word.
        snap = re.search(r"[\.\?!]\s+(.*)", prefix, flags=re.DOTALL)
        if snap:
            prefix = snap.group(1).strip()
    return _build_mc_messages(prefix, last_word), {
        "record_id": record.get("id") or record.get("story_id"),
        "rationale": source_cfg.rationale,
    }


def _normalize_cbt_record(
    record: Mapping[str, Any], source_cfg: SFTSourceConfig
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Render CBT examples as LAMBADA-shape (passage_minus_cloze, cloze_answer).

    The HF `cbt` schema provides:
      - sentences: 20 sentences of context
      - question: a 21st sentence with the cloze position marked 'XXXXX'
      - answer: the cloze word (typically a character name)

    We build the prompt as <context> + <question_prefix_before_XXXXX> with a
    trailing space, and use `answer` itself as the assistant target. Anything
    after XXXXX in the question is discarded — the model learns to predict
    the cloze answer from the long-range context, exactly as LAMBADA tests.
    """
    sentences = record.get("sentences")
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    if not isinstance(sentences, list) or not sentences or not question or not answer:
        return None

    if "XXXXX" not in question:
        return None
    head, _tail = question.split("XXXXX", 1)

    context = " ".join(str(s).strip() for s in sentences if str(s).strip())
    if not context:
        return None

    # Trailing space mirrors the LAMBADA benchmark prompt shape.
    prompt = f"{context} {head.rstrip()} "

    cleaned_answer = clean_message_content(answer).strip(_LAMBADA_STRIP_CHARS)
    if not cleaned_answer:
        return None

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": cleaned_answer.lower()},
    ]
    metadata = {"rationale": source_cfg.rationale, "kind": "lambada_shape"}
    return messages, metadata


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

    words = text.split()
    if len(words) < 30:
        return None

    last_word = words[-1].strip(_LAMBADA_STRIP_CHARS)
    # Reject targets that are pure punctuation or contain digits (LAMBADA-style
    # targets are content words; numbers/dates would distort the regularizer).
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


_FLAN_MC_QUESTION_PATTERN = re.compile(
    r"(?:^|\n)\s*[A-D]\)\s+\S",  # detects "A) ...", "B) ..."
)
_FLAN_MC_RESPONSE_LETTER_PATTERN = re.compile(
    r"(?:answer\s*(?:is|:)\s*|^\s*)([A-D])\b",
    re.IGNORECASE,
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


def _normalize_openbookqa_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    # allenai/openbookqa fields: question_stem, choices={text, label}, answerKey, id.
    stem = str(record.get("question_stem") or "").strip()
    raw_choices = record.get("choices")
    answer_key = str(record.get("answerKey") or "").strip().upper()
    if not stem or not isinstance(raw_choices, Mapping) or not answer_key:
        return None
    texts = raw_choices.get("text") or []
    labels = raw_choices.get("label") or []
    if not isinstance(texts, list) or len(texts) != 4 or not isinstance(labels, list) or len(labels) != 4:
        return None
    cleaned = [str(t).strip() for t in texts]
    if any(not c for c in cleaned):
        return None
    try:
        answer_index = [str(l).strip().upper() for l in labels].index(answer_key)
    except ValueError:
        return None
    seed_key = str(record.get("id") or stem)
    cleaned, answer_index = _permute_choices(cleaned, answer_index, seed_key)
    prompt = _format_mc_prompt(stem, cleaned, prefix="Question")
    return _build_mc_messages(prompt, "ABCD"[answer_index]), {
        "record_id": record.get("id"),
        "rationale": source_cfg.rationale,
    }


def _normalize_winogrande_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    sentence = str(record.get("sentence") or "").strip()
    option1 = str(record.get("option1") or "").strip()
    option2 = str(record.get("option2") or "").strip()
    answer = record.get("answer")
    if not sentence or not option1 or not option2 or answer in (None, ""):
        return None
    try:
        answer_index = int(answer) - 1  # WinoGrande uses 1-based labels
    except (TypeError, ValueError):
        return None
    if answer_index not in (0, 1):
        return None
    options = [option1, option2]
    seed_key = str(record.get("qID") or sentence[:64])
    options, answer_index = _permute_choices(options, answer_index, seed_key)
    prompt = _format_mc_prompt(sentence, options, prefix="Context")
    answer_letter = "AB"[answer_index]
    return _build_mc_messages(prompt, answer_letter), {
        "record_id": record.get("qID"),
        "rationale": source_cfg.rationale,
    }


def _normalize_wildchat_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    model_name = str(record.get("model") or record.get("model_name") or "").lower()
    if source_cfg.require_model_substring and source_cfg.require_model_substring.lower() not in model_name:
        return None
    if source_cfg.exclude_toxic and (
        _boolish(record.get("toxic"))
        or _boolish(record.get("toxicity"))
        or _boolish(record.get("flagged"))
    ):
        return None
    if source_cfg.exclude_redacted and (
        _boolish(record.get("redacted"))
        or _boolish(record.get("is_redacted"))
    ):
        return None
    language = str(record.get("language") or record.get("lang") or "").lower()
    if source_cfg.language and language and not language.startswith(source_cfg.language.lower()):
        return None
    raw_conversation = record.get("conversation") or record.get("messages") or record.get("turns") or []
    if not isinstance(raw_conversation, list):
        return None
    messages: list[dict[str, str]] = []
    for turn in raw_conversation:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("role") or turn.get("from") or turn.get("speaker") or "")
        content = str(turn.get("content") or turn.get("text") or turn.get("message") or "")
        messages.append({"role": role, "content": content})
    metadata = {
        "conversation_id": record.get("conversation_id") or record.get("id"),
        "model": record.get("model") or record.get("model_name"),
    }
    return messages, metadata


def _oasst_rank_key(record: Mapping[str, Any]) -> tuple[float, float, str]:
    rank = record.get("rank")
    review_count = record.get("review_count")
    score = record.get("score")
    try:
        rank_value = float(rank)
    except Exception:
        rank_value = math.inf
    try:
        review_value = -float(review_count)
    except Exception:
        review_value = 0.0
    try:
        score_value = -float(score)
    except Exception:
        score_value = 0.0
    return rank_value, review_value + score_value, str(record.get("message_id", ""))


def _collect_oasst_branches(
    source_cfg: SFTSourceConfig,
    *,
    snapshot_path: Path | None = None,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> list[tuple[list[dict[str, str]], dict[str, Any]]]:
    rows = list(
        _load_records(
            source_cfg,
            snapshot_path=snapshot_path,
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
    )
    by_parent: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    candidates: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        lang = str(row.get("lang") or row.get("language") or "").lower()
        if source_cfg.language and lang and lang != source_cfg.language.lower():
            continue
        tree_state = row.get("tree_state")
        if source_cfg.require_tree_state and tree_state != source_cfg.require_tree_state:
            continue
        if _boolish(row.get("deleted")):
            continue
        message_id = row.get("message_id")
        if not message_id:
            continue
        candidates[str(message_id)] = row
        parent_id = row.get("parent_id")
        by_parent[str(parent_id) if parent_id is not None else None].append(row)

    roots = sorted(
        by_parent.get(None, []),
        key=lambda item: str(item.get("message_id", "")),
    )
    branches: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
    for root in roots:
        role = str(root.get("role") or "")
        if role.lower() not in {"prompter", "user", "human"}:
            continue
        branch = [root]
        current = root
        while len(branch) < source_cfg.max_turns:
            children = by_parent.get(str(current.get("message_id")), [])
            if not children:
                break
            if source_cfg.use_best_branch:
                child = sorted(children, key=_oasst_rank_key)[0]
            else:
                child = sorted(children, key=lambda item: str(item.get("message_id", "")))[0]
            branch.append(child)
            current = child
        messages = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("text") or item.get("content") or ""),
            }
            for item in branch
        ]
        metadata = {
            "message_tree_id": root.get("message_tree_id"),
            "root_id": root.get("message_id"),
        }
        branches.append((messages, metadata))
    return branches


def _normalize_tulu_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    source_name = str(record.get("source") or record.get("dataset") or "").lower()
    if source_cfg.source_matches:
        normalized_source_name = _normalize_source_key(source_name)
        if not any(
            match.lower() in source_name
            or _normalize_source_key(match) in normalized_source_name
            for match in source_cfg.source_matches
        ):
            return None
    raw_messages = record.get("messages") or record.get("conversation") or []
    if not isinstance(raw_messages, list):
        return None
    messages = []
    for item in raw_messages:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or item.get("from") or "")
        content = str(item.get("content") or item.get("text") or "")
        messages.append({"role": role, "content": content})
    metadata = {"source_name": source_name}
    return messages, metadata


def _normalize_wildguardmix_record(record: Mapping[str, Any], source_cfg: SFTSourceConfig) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    if source_cfg.keep_harmful_only:
        harmful = (
            _boolish(record.get("harmful"))
            or str(record.get("prompt_harm_label") or record.get("prompt_label") or "").lower()
            in {"harmful", "unsafe", "toxic"}
        )
        refusal = (
            _boolish(record.get("refusal"))
            or str(record.get("response_refusal_label") or record.get("response_label") or "").lower()
            in {"refusal", "safe_refusal", "refuse"}
        )
        if not harmful or not refusal:
            return None
    prompt = str(record.get("prompt") or record.get("instruction") or "").strip()
    response = str(
        record.get("response")
        or record.get("chosen")
        or record.get("assistant_response")
        or record.get("safe_response")
        or ""
    ).strip()
    if not prompt or not response:
        return None
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    metadata = {"source_row_id": record.get("id")}
    return messages, metadata


def _normalize_pku_safe_rlhf_qa_record(
    record: Mapping[str, Any],
    source_cfg: SFTSourceConfig,
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    del source_cfg  # PKU-SafeRLHF-QA is already a safety-focused dataset; keep safe rows directly.
    prompt = str(record.get("prompt") or "").strip()
    response = str(record.get("response") or "").strip()
    if not prompt or not response:
        return None

    is_safe = _boolish(record.get("is_safe"))
    if not is_safe:
        return None

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    metadata = {
        "source_row_id": record.get("sha256"),
        "prompt_source": record.get("prompt_source"),
        "response_source": record.get("response_source"),
        "severity_level": record.get("severity_level"),
    }
    return messages, metadata


def _normalize_source_record(
    record: Mapping[str, Any],
    *,
    source_cfg: SFTSourceConfig,
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    if source_cfg.loader == "local_jsonl":
        return _normalize_local_jsonl_record(record, source_cfg)
    if source_cfg.loader == "alpaca":
        return _normalize_alpaca_record(record, source_cfg)
    if source_cfg.loader == "ai2_arc":
        return _normalize_ai2_arc_record(record, source_cfg)
    if source_cfg.loader == "wildchat":
        return _normalize_wildchat_record(record, source_cfg)
    if source_cfg.loader == "tulu":
        return _normalize_tulu_record(record, source_cfg)
    if source_cfg.loader == "wildguardmix":
        return _normalize_wildguardmix_record(record, source_cfg)
    if source_cfg.loader == "pku_safe_rlhf_qa":
        return _normalize_pku_safe_rlhf_qa_record(record, source_cfg)
    if source_cfg.loader == "sciq":
        return _normalize_sciq_record(record, source_cfg)
    if source_cfg.loader == "commonsense_qa":
        return _normalize_commonsense_qa_record(record, source_cfg)
    if source_cfg.loader == "cosmos_qa":
        return _normalize_cosmos_qa_record(record, source_cfg)
    if source_cfg.loader == "social_iqa":
        return _normalize_social_iqa_record(record, source_cfg)
    if source_cfg.loader == "race":
        return _normalize_race_record(record, source_cfg)
    if source_cfg.loader == "mmlu":
        return _normalize_mmlu_record(record, source_cfg)
    if source_cfg.loader == "boolq":
        return _normalize_boolq_record(record, source_cfg)
    if source_cfg.loader == "piqa":
        return _normalize_piqa_record(record, source_cfg)
    if source_cfg.loader == "wsc273":
        return _normalize_wsc273_record(record, source_cfg)
    if source_cfg.loader == "hellaswag":
        return _normalize_hellaswag_record(record, source_cfg)
    if source_cfg.loader == "winogrande":
        return _normalize_winogrande_record(record, source_cfg)
    if source_cfg.loader == "openbookqa":
        return _normalize_openbookqa_record(record, source_cfg)
    if source_cfg.loader == "narrative_completion":
        return _normalize_narrative_completion_record(record, source_cfg)
    if source_cfg.loader == "cbt":
        return _normalize_cbt_record(record, source_cfg)
    if source_cfg.loader == "bookcorpus_lambada":
        return _normalize_bookcorpus_lambada_record(record, source_cfg)
    if source_cfg.loader == "flan_mc":
        return _normalize_flan_mc_record(record, source_cfg)
    raise ValueError(f"Unsupported loader for record-level normalization: {source_cfg.loader}")


def _prepare_message_example(
    messages: list[dict[str, str]],
    *,
    source_cfg: SFTSourceConfig,
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
    lang_filter: OptionalLanguageFilter,
    metadata: dict[str, Any],
    template_format: str = "alpaca",
) -> PreparedExample | None:
    messages = _sanitize_messages(messages)
    if not messages:
        return None
    if source_cfg.drop_chain_of_thought and any(
        _looks_like_chain_of_thought(message["content"])
        for message in messages
        if message["role"] == "assistant"
    ):
        return None
    prompt_text = _source_prompt_text(messages)
    if not prompt_text:
        return None
    if not lang_filter.matches(prompt_text, source_cfg.language):
        return None
    tokenized = trim_messages_to_token_limit(
        tokenizer,
        messages,
        system_prompt=system_prompt,
        max_tokens=max_seq_length + 1,
        append_eos=True,
        template_format=template_format,
    )
    if tokenized is None:
        return None
    if len(tokenized.messages) < source_cfg.min_turns:
        return None
    if len(tokenized.messages) > source_cfg.max_turns + 1:
        return None
    if sum(tokenized.token_loss_mask) == 0:
        return None
    quality = _quality_score(tokenized.messages, source_cfg=source_cfg)
    rendered_text = "\n\n".join(
        f"{item['role']}: {item['content']}" for item in tokenized.messages
    )
    return PreparedExample(
        source=source_cfg.name,
        tags=list(source_cfg.tags),
        quality_score=quality,
        prompt_hash=_stable_hash(_normalize_text(prompt_text)),
        prompt_text=prompt_text,
        full_text_hash=_stable_hash(_normalize_text(rendered_text)),
        tokens=list(tokenized.tokens),
        loss_mask=list(tokenized.token_loss_mask),
        messages=tokenized.messages,
        metadata=metadata,
    )


def _collect_candidates_for_source(
    source_cfg: SFTSourceConfig,
    *,
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
    lang_filter: OptionalLanguageFilter,
    snapshot_path: Path | None = None,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
    template_format: str = "alpaca",
) -> list[PreparedExample]:
    target_candidates = source_cfg.target_examples * source_cfg.candidate_multiplier
    prepared: list[PreparedExample] = []
    if source_cfg.loader == "oasst1":
        branches = _collect_oasst_branches(
            source_cfg,
            snapshot_path=snapshot_path,
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
        iterable: Iterable[tuple[list[dict[str, str]], dict[str, Any]]] = branches
    else:
        iterable = []
        raw_records = _load_records(
            source_cfg,
            snapshot_path=snapshot_path,
            cache_dir=cache_dir,
            hf_token=hf_token,
        )
        iterable = []
        for record in raw_records:
            normalized = _normalize_source_record(record, source_cfg=source_cfg)
            if normalized is not None:
                iterable.append(normalized)

    seen_full_hashes: set[str] = set()
    for messages, metadata in iterable:
        item = _prepare_message_example(
            messages,
            source_cfg=source_cfg,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_seq_length=max_seq_length,
            lang_filter=lang_filter,
            metadata=metadata,
            template_format=template_format,
        )
        if item is None:
            continue
        if item.full_text_hash in seen_full_hashes:
            continue
        seen_full_hashes.add(item.full_text_hash)
        prepared.append(item)
        if len(prepared) >= target_candidates:
            break

    prepared.sort(key=lambda item: (-item.quality_score, item.prompt_hash))
    return prepared


def _iter_local_disk_texts(path: Path, field: str | None) -> Iterator[str]:
    if path.is_file():
        if path.suffix == ".jsonl":
            for row in _iter_local_jsonl(path):
                value = row.get(field) if field else row.get("text")
                if value:
                    yield str(value)
        else:
            yield path.read_text(encoding="utf-8", errors="ignore")
        return

    try:
        ds = load_from_disk(str(path))
        for row in ds:
            value = row.get(field) if field else row.get("text")
            if value:
                yield str(value)
        return
    except Exception:
        pass

    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix == ".jsonl":
            yield from _iter_local_disk_texts(candidate, field)
            continue
        if candidate.suffix in {".txt", ".md"}:
            yield candidate.read_text(encoding="utf-8", errors="ignore")


def _extract_hf_prompt(record: Mapping[str, Any], dataset_name: str, field: str | None) -> str | None:
    if field and record.get(field):
        return str(record[field])
    if dataset_name == "hellaswag":
        if record.get("ctx"):
            return str(record["ctx"])
        ctx_a = record.get("ctx_a")
        ctx_b = record.get("ctx_b")
        if ctx_a or ctx_b:
            return " ".join(part for part in [ctx_a, ctx_b] if part)
    if dataset_name == "winogrande":
        return str(record.get("sentence") or "")
    if dataset_name == "openbookqa":
        return str(record.get("question_stem") or "")
    if dataset_name == "lambada":
        return str(record.get("text") or "")
    return None


def _build_decontam_index(
    configs: list[SFTDecontamConfig],
    *,
    snapshot_paths: Mapping[str, Path] | None = None,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> PromptContaminationIndex:
    index = PromptContaminationIndex()
    snapshot_paths = snapshot_paths or {}
    for cfg in configs:
        if not cfg.enabled:
            continue
        try:
            snapshot_path = snapshot_paths.get(cfg.name)
            if snapshot_path is not None:
                if not snapshot_path.exists():
                    raise FileNotFoundError(
                        f"HF decontamination snapshot for '{cfg.name}' is missing at {snapshot_path}. "
                        "Run `--stage sft-download` first."
                    )
                for text in _iter_local_disk_texts(snapshot_path, cfg.field):
                    if text:
                        index.add(text)
                continue
            if cfg.loader == "local_disk":
                for text in _iter_local_disk_texts(Path(cfg.path), cfg.field):
                    if text:
                        index.add(text)
            else:
                load_kwargs: dict[str, Any] = {"split": cfg.split}
                if cache_dir is not None:
                    load_kwargs["cache_dir"] = str(cache_dir)
                if hf_token is not None:
                    load_kwargs["token"] = hf_token
                dataset = load_dataset(cfg.path, cfg.subset, **load_kwargs)
                for row in dataset:
                    text = _extract_hf_prompt(row, cfg.name, cfg.field)
                    if text:
                        index.add(text)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - depends on network/local files
            log.warning("Skipping decontamination source %s: %s", cfg.name, exc)
    return index


def _select_final_examples(
    source_candidates: dict[str, list[PreparedExample]],
    *,
    source_cfgs: list[SFTSourceConfig],
    decontam_index: PromptContaminationIndex,
) -> tuple[list[PreparedExample], dict[str, dict[str, int]]]:
    selected: list[PreparedExample] = []
    stats: dict[str, dict[str, int]] = {}
    seen_prompts: set[str] = set()
    accepted_index = PromptContaminationIndex()

    for cfg in source_cfgs:
        kept = 0
        dropped_exact = 0
        dropped_fuzzy = 0
        dropped_contam = 0
        for candidate in source_candidates.get(cfg.name, []):
            if kept >= cfg.target_examples:
                break
            if candidate.prompt_hash in seen_prompts:
                dropped_exact += 1
                continue
            if decontam_index.contains(candidate.prompt_text):
                dropped_contam += 1
                continue
            if accepted_index.contains(candidate.prompt_text):
                dropped_fuzzy += 1
                continue
            accepted_index.add(candidate.prompt_text)
            seen_prompts.add(candidate.prompt_hash)
            selected.append(candidate)
            kept += 1
        stats[cfg.name] = {
            "target": cfg.target_examples,
            "kept": kept,
            "dropped_exact": dropped_exact,
            "dropped_fuzzy": dropped_fuzzy,
            "dropped_contam": dropped_contam,
            "available_candidates": len(source_candidates.get(cfg.name, [])),
        }
    return selected, stats


def _split_examples(examples: list[PreparedExample]) -> dict[str, list[PreparedExample]]:
    by_source: dict[str, list[PreparedExample]] = defaultdict(list)
    for example in examples:
        by_source[example.source].append(example)

    splits = {"train": [], "dev": [], "test": []}
    for source, items in by_source.items():
        del source  # source kept only for grouping clarity
        ordered = sorted(items, key=lambda item: item.prompt_hash)
        total = len(ordered)
        dev_n = max(1, round(total * 0.05)) if total >= 20 else max(1, total // 10) if total >= 3 else 0
        test_n = max(1, round(total * 0.05)) if total >= 20 else max(1, total // 10) if total >= 3 else 0
        train_n = max(0, total - dev_n - test_n)
        splits["train"].extend(ordered[:train_n])
        splits["dev"].extend(ordered[train_n : train_n + dev_n])
        splits["test"].extend(ordered[train_n + dev_n :])
    for name in splits:
        splits[name].sort(key=lambda item: (item.source, item.prompt_hash))
    return splits


def _pack_examples(examples: list[PreparedExample], *, max_seq_length: int) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current_tokens: list[int] = []
    current_mask: list[int] = []
    source_counts: dict[str, int] = defaultdict(int)
    quality_sum = 0.0
    for example in examples:
        if len(example.tokens) > max_seq_length + 1:
            continue
        if current_tokens and len(current_tokens) + len(example.tokens) > max_seq_length + 1:
            packed.append(
                {
                    "tokens": current_tokens,
                    "loss_mask": current_mask,
                    "quality_score": quality_sum / max(1, sum(source_counts.values())),
                    "source_counts": dict(source_counts),
                    "num_examples": sum(source_counts.values()),
                }
            )
            current_tokens = []
            current_mask = []
            source_counts = defaultdict(int)
            quality_sum = 0.0
        current_tokens.extend(example.tokens)
        current_mask.extend(example.loss_mask)
        source_counts[example.source] += 1
        quality_sum += example.quality_score

    if current_tokens:
        packed.append(
            {
                "tokens": current_tokens,
                "loss_mask": current_mask,
                "quality_score": quality_sum / max(1, sum(source_counts.values())),
                "source_counts": dict(source_counts),
                "num_examples": sum(source_counts.values()),
            }
        )
    return packed


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def _cleanup_cache_dir(cache_dir: Path | None, *, protected_paths: list[Path]) -> bool:
    if cache_dir is None or not cache_dir.exists():
        return False

    resolved_cache = cache_dir.resolve()
    for protected_path in protected_paths:
        resolved_protected = protected_path.resolve()
        if resolved_protected == resolved_cache:
            log.warning(
                "Skipping SFT cache cleanup because protected path %s equals cache dir %s",
                resolved_protected,
                resolved_cache,
            )
            return False
        if os.path.commonpath([str(resolved_cache), str(resolved_protected)]) == str(resolved_cache):
            log.warning(
                "Skipping SFT cache cleanup because protected path %s lives inside cache dir %s",
                resolved_protected,
                resolved_cache,
            )
            return False

    shutil.rmtree(resolved_cache)
    log.info("Removed SFT cache directory: %s", resolved_cache)
    return True


def _serialize_example(example: PreparedExample) -> dict[str, Any]:
    payload = asdict(example)
    return payload


def run_prepare_sft(
    project_config: ProjectConfig,
    *,
    seed: int | None = None,
    hf_token: str | None = None,
) -> dict[str, str]:
    sft_cfg = project_config.sft
    if sft_cfg is None:
        raise ValueError("SFT configuration missing; cannot prepare posttraining data.")

    effective_seed = int(sft_cfg.seed if seed is None else seed)
    np.random.seed(effective_seed)
    prepared_dir = Path(sft_cfg.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(sft_cfg.raw_dir)

    tokenizer = build_tokenizer()
    lang_filter = OptionalLanguageFilter()
    log.info("Preparing SFT data into %s", prepared_dir)

    source_candidates: dict[str, list[PreparedExample]] = {}
    for source_cfg in sft_cfg.sources:
        snapshot_path = get_source_snapshot_path(raw_dir, source_cfg)
        # Per-source template_format wins over the global default. Lets a
        # narrative-completion source train as raw text (LAMBADA shape) while
        # the rest of the recipe stays Alpaca-wrapped.
        effective_template = source_cfg.template_format or sft_cfg.template_format
        candidates = _collect_candidates_for_source(
            source_cfg,
            tokenizer=tokenizer,
            system_prompt=sft_cfg.system_prompt,
            max_seq_length=sft_cfg.max_seq_length,
            lang_filter=lang_filter,
            snapshot_path=snapshot_path,
            template_format=effective_template,
        )
        source_candidates[source_cfg.name] = candidates
        log.info("Collected %d candidates for %s", len(candidates), source_cfg.name)

    decontam_snapshot_paths = {
        cfg.name: snapshot_path
        for cfg in sft_cfg.decontam_datasets
        if (snapshot_path := get_decontam_snapshot_path(raw_dir, cfg)) is not None
    }
    decontam_index = _build_decontam_index(
        sft_cfg.decontam_datasets,
        snapshot_paths=decontam_snapshot_paths,
    )
    selected, source_stats = _select_final_examples(
        source_candidates,
        source_cfgs=sft_cfg.sources,
        decontam_index=decontam_index,
    )
    splits = _split_examples(selected)

    train_examples_path = prepared_dir / "train_examples.jsonl"
    dev_examples_path = prepared_dir / "dev_examples.jsonl"
    test_examples_path = prepared_dir / "test_examples.jsonl"
    train_packed_path = prepared_dir / "train_packed.jsonl"
    dev_packed_path = prepared_dir / "dev_packed.jsonl"
    test_packed_path = prepared_dir / "test_packed.jsonl"

    _write_jsonl(train_examples_path, (_serialize_example(item) for item in splits["train"]))
    _write_jsonl(dev_examples_path, (_serialize_example(item) for item in splits["dev"]))
    _write_jsonl(test_examples_path, (_serialize_example(item) for item in splits["test"]))

    train_packed = _pack_examples(splits["train"], max_seq_length=sft_cfg.max_seq_length)
    dev_packed = _pack_examples(splits["dev"], max_seq_length=sft_cfg.max_seq_length)
    test_packed = _pack_examples(splits["test"], max_seq_length=sft_cfg.max_seq_length)
    _write_jsonl(train_packed_path, train_packed)
    _write_jsonl(dev_packed_path, dev_packed)
    _write_jsonl(test_packed_path, test_packed)

    manifest = {
        "seed": effective_seed,
        "system_prompt": sft_cfg.system_prompt,
        "max_seq_length": sft_cfg.max_seq_length,
        "raw_dir": str(raw_dir),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "packed_counts": {
            "train": len(train_packed),
            "dev": len(dev_packed),
            "test": len(test_packed),
        },
        "source_stats": source_stats,
        "train_examples_path": str(train_examples_path),
        "dev_examples_path": str(dev_examples_path),
        "test_examples_path": str(test_examples_path),
        "train_packed_path": str(train_packed_path),
        "dev_packed_path": str(dev_packed_path),
        "test_packed_path": str(test_packed_path),
        "sources": [
            {
                "name": cfg.name,
                "target_examples": cfg.target_examples,
                "tags": cfg.tags,
                "rationale": cfg.rationale,
            }
            for cfg in sft_cfg.sources
        ],
    }
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Prepared SFT dataset manifest -> %s", manifest_path)
    return {
        "manifest": str(manifest_path),
        "train_packed": str(train_packed_path),
        "dev_packed": str(dev_packed_path),
        "test_packed": str(test_packed_path),
    }
