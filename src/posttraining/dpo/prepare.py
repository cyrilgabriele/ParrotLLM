"""Prepare DPO preference pairs from HH-RLHF (or compatible) into packed JSONL.

Output records have the contract:
  {
    "prompt_tokens": [...],
    "chosen_tokens":  [...prompt_tokens..., ...response_chosen_tokens...],
    "rejected_tokens":[...prompt_tokens..., ...response_rejected_tokens...],
    "prompt_len": int   # so trainer can slice prompt vs response
  }
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from configs import ProjectConfig
from src.posttraining.templates import normalize_messages, render_conversation
from src.utils import build_tokenizer


log = logging.getLogger("parrotllm.posttraining.dpo")


@dataclass(slots=True)
class PreparedPreferencePair:
    prompt: str
    chosen: str
    rejected: str


def _split_final(text: str) -> tuple[str, str] | None:
    """Return (final_user_prompt, final_assistant_response) from an HH-RLHF turn string.

    HH-RLHF stores entire conversations as repeating
    "\\n\\nHuman: ...\\n\\nAssistant: ..." segments. We keep only the FINAL
    user/assistant pair as (prompt, response).
    """
    a_marker = "\n\nAssistant:"
    h_marker = "\n\nHuman:"
    last_a = text.rfind(a_marker)
    if last_a < 0:
        return None
    final_response = text[last_a + len(a_marker):].strip()
    prefix = text[:last_a]
    last_h = prefix.rfind(h_marker)
    if last_h < 0:
        return None
    final_prompt = prefix[last_h + len(h_marker):].strip()
    if not final_prompt or not final_response:
        return None
    return final_prompt, final_response


def parse_hh_rlhf_record(raw: dict[str, Any]) -> PreparedPreferencePair | None:
    """Extract the final user prompt + chosen/rejected assistant responses from an HH-RLHF row."""
    chosen = raw.get("chosen", "")
    rejected = raw.get("rejected", "")
    if not chosen or not rejected or chosen.strip() == rejected.strip():
        return None

    chosen_split = _split_final(chosen)
    rejected_split = _split_final(rejected)
    if chosen_split is None or rejected_split is None:
        return None

    prompt_c, resp_c = chosen_split
    prompt_r, resp_r = rejected_split
    # Must share the same prompt; otherwise this isn't a valid pair.
    if prompt_c.strip() != prompt_r.strip():
        return None
    if not resp_c.strip() or not resp_r.strip() or resp_c.strip() == resp_r.strip():
        return None
    return PreparedPreferencePair(prompt=prompt_c, chosen=resp_c, rejected=resp_r)


def pack_pair(
    pair: PreparedPreferencePair,
    *,
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
) -> dict[str, Any]:
    """Render via the Alpaca template, tokenize, and produce the packed record."""
    prompt_messages = normalize_messages(
        [{"role": "user", "content": pair.prompt}],
        system_prompt=system_prompt,
        require_final_assistant=False,
    )
    prompt_render = render_conversation(prompt_messages, add_generation_prompt=True)
    prompt_text = prompt_render.text  # already ends with "\n\n### Response:\n"

    chosen_text = prompt_text + pair.chosen.strip()
    rejected_text = prompt_text + pair.rejected.strip()

    prompt_tokens = tokenizer.encode(prompt_text)
    chosen_tokens = tokenizer.encode(chosen_text)
    rejected_tokens = tokenizer.encode(rejected_text)

    if len(chosen_tokens) > max_seq_length or len(rejected_tokens) > max_seq_length:
        chosen_tokens = chosen_tokens[:max_seq_length]
        rejected_tokens = rejected_tokens[:max_seq_length]

    return {
        "prompt_tokens": prompt_tokens,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "prompt_len": len(prompt_tokens),
    }


def _decontam_set(decontam_specs: Iterable, tokenizer) -> set[str]:
    """Legacy HH-RLHF stub: returns empty set.

    Kept for the HH-RLHF path which uses sha1(prompt) hash blocklist semantics.
    The real eval-split decontam for the MC-letter path lives in
    `_build_dpo_decontam_index` + `_filter_decontaminated`, which use the
    same MinHash + 5-gram + Jaccard 0.8 machinery as the SFT side.
    """
    specs_list = list(decontam_specs)
    if specs_list:
        log.info(
            "DPO HH-RLHF decontam stub active (%d decontam_datasets entries ignored). "
            "MC-letter path uses _build_dpo_decontam_index instead.",
            len(specs_list),
        )
    return set()


def _build_dpo_decontam_index(decontam_specs: Iterable):
    """Build a `PromptContaminationIndex` from raw DPO decontam_datasets entries.

    `dpo.decontam_datasets` is typed as `list` (raw dicts) in the config, so we
    coerce each entry into an `SFTDecontamConfig` before delegating to the
    SFT-side `_build_decontam_index`. DPO inherits the SFT-side decontam config
    semantics because both protect the same eval splits
    (HellaSwag/WinoGrande/OBQA/LAMBADA + Wikitext-103/OWT-eval).
    """
    from configs.posttraining.sftConfig import SFTDecontamConfig
    from src.posttraining.prepare import PromptContaminationIndex, _build_decontam_index

    parsed: list[SFTDecontamConfig] = []
    for spec in decontam_specs:
        if isinstance(spec, SFTDecontamConfig):
            parsed.append(spec)
        elif isinstance(spec, dict):
            parsed.append(SFTDecontamConfig.model_validate(spec))
        else:
            # Pydantic-shaped object with the right fields; try to coerce via dict view.
            try:
                parsed.append(SFTDecontamConfig.model_validate(dict(spec)))
            except Exception as exc:  # pragma: no cover - defensive path
                log.warning("Skipping unrecognized DPO decontam spec %r: %s", spec, exc)
    if not parsed:
        return PromptContaminationIndex()  # empty index, contains() always False
    return _build_decontam_index(parsed)


def _filter_decontaminated(
    pairs: list[dict[str, Any]],
    decontam_prompts,  # set[str] for cheap exact-substring check, OR PromptContaminationIndex
) -> list[dict[str, Any]]:
    """Drop any pair whose prompt overlaps an eval-split entry.

    Accepts either:
      - a flat `set[str]` of normalized prompt strings (cheap substring check), or
      - a `PromptContaminationIndex` (MinHash + 5-gram + Jaccard 0.8).
    """
    kept: list[dict[str, Any]] = []
    for pair in pairs:
        prompt = pair.get("prompt") or ""
        if isinstance(decontam_prompts, set):
            normalized = " ".join(prompt.lower().split())
            if any(entry in normalized for entry in decontam_prompts):
                continue
        else:
            # Duck-typed: PromptContaminationIndex.contains(text) -> bool
            if decontam_prompts.contains(prompt):
                continue
        kept.append(pair)
    return kept


def run_prepare_dpo(project_config: ProjectConfig, *, seed: int, hf_token: str | None = None) -> None:
    """Pipeline entry point: dispatch on `dpo.preference_format`."""
    dpo = project_config.dpo
    if dpo is None:
        raise ValueError("project config has no `dpo` section")
    if dpo.preference_format == "mc_letter":
        return _run_prepare_dpo_mc_letter(project_config, seed=seed, hf_token=hf_token)
    return _run_prepare_dpo_hh_rlhf(project_config, seed=seed, hf_token=hf_token)


def _run_prepare_dpo_hh_rlhf(project_config: ProjectConfig, *, seed: int, hf_token: str | None = None) -> None:
    """Original HH-RLHF preparation path."""
    from datasets import load_dataset

    dpo = project_config.dpo
    if dpo is None:
        raise ValueError("project config has no `dpo` section")

    out_dir = Path(dpo.prepared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    tokenizer = build_tokenizer()
    decontam_blocklist = _decontam_set(dpo.decontam_datasets, tokenizer)

    n_pairs = 0
    n_train = 0
    n_dev = 0
    n_dropped_decontam = 0
    n_dropped_invalid = 0
    target = sum(s.target_pairs for s in dpo.sources)
    train_target = max(0, target - dpo.dev_pairs)

    with train_path.open("w", encoding="utf-8") as f_train, dev_path.open("w", encoding="utf-8") as f_dev:
        for source in dpo.sources:
            log.info("Loading source: %s (%s)", source.name, source.path)
            ds = load_dataset(
                source.path,
                data_dir=source.subset,
                split=source.split,
                cache_dir=str(dpo.cache_dir) if dpo.cache_dir else None,
                token=hf_token,
            )
            for raw in ds:
                if n_pairs >= target:
                    break
                parsed = parse_hh_rlhf_record(raw)
                if parsed is None:
                    n_dropped_invalid += 1
                    continue
                prompt_hash = hashlib.sha1(parsed.prompt.strip().encode("utf-8")).hexdigest()
                if prompt_hash in decontam_blocklist:
                    n_dropped_decontam += 1
                    continue
                packed = pack_pair(
                    parsed,
                    tokenizer=tokenizer,
                    system_prompt=dpo.system_prompt,
                    max_seq_length=dpo.max_seq_length,
                )
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
        "preference_format": "hh_rlhf",
        "n_train": n_train,
        "n_dev": n_dev,
        "n_dropped_decontam": n_dropped_decontam,
        "n_dropped_invalid": n_dropped_invalid,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("DPO prepare finished: %s", manifest)


def _build_letter_dpo_pair(
    *,
    user_prompt: str,
    correct_letter: str,
    candidate_letters: list[str],
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Build a (prompt, chosen=correct, rejected=wrong) packed JSONL record.

    Mirrors `pack_pair`'s output shape so the existing DPO trainer consumes it
    unchanged. The chosen response is the correct letter; the rejected
    response is a randomly-sampled wrong letter from the same MC option set.
    """
    wrong_options = [letter for letter in candidate_letters if letter != correct_letter]
    if not wrong_options:
        return None
    rejected_letter = rng.choice(wrong_options)

    prompt_messages = normalize_messages(
        [{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        require_final_assistant=False,
    )
    prompt_render = render_conversation(prompt_messages, add_generation_prompt=True)
    prompt_text = prompt_render.text  # ends with "\n\n### Response:\n"

    chosen_text = prompt_text + correct_letter
    rejected_text = prompt_text + rejected_letter

    prompt_tokens = tokenizer.encode(prompt_text)
    chosen_tokens = tokenizer.encode(chosen_text)
    rejected_tokens = tokenizer.encode(rejected_text)

    if len(chosen_tokens) > max_seq_length or len(rejected_tokens) > max_seq_length:
        # Skip; safer than truncating a 1-token response (would change loss).
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "prompt_len": len(prompt_tokens),
    }


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

    Mirrors `_build_letter_dpo_pair`: same return shape so the existing DPO trainer
    consumes it unchanged. The chosen response is the correct continuation text
    (full text, not just a letter); the rejected response is a randomly-sampled
    distractor continuation from the same MC option set.
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


def _run_prepare_dpo_mc_letter(
    project_config: ProjectConfig,
    *,
    seed: int,
    hf_token: str | None = None,
) -> None:
    """Build (correct-letter, wrong-letter) preference pairs from MC sources.

    Reuses the SFT MC normalizers via `_normalize_source_record`: each source
    must declare a `loader` field naming an SFT MC loader (e.g. "piqa", "sciq").
    The normalizer yields a 2-message conversation [user_prompt, correct_letter];
    we then sample a wrong letter from the same option set as the rejected
    response. Output records share the HH-RLHF `pack_pair` shape, so the
    existing DPO trainer consumes them unchanged.
    """
    from datasets import load_dataset

    from configs.posttraining.sftConfig import SFTSourceConfig
    from src.posttraining.prepare import _normalize_source_record

    dpo = project_config.dpo
    if dpo is None:
        raise ValueError("project config has no `dpo` section")
    if not dpo.sources:
        raise ValueError("dpo.sources is empty; need at least one MC source")

    out_dir = Path(dpo.prepared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    tokenizer = build_tokenizer()
    rng = random.Random(int(seed))

    # Build the eval-split contamination index (same MinHash + 5-gram + Jaccard 0.8
    # machinery the SFT pipeline uses). MC-letter prompts come from
    # HellaSwag/WinoGrande/OBQA-train/MMLU-aux, where overlap with the matching
    # eval splits IS plausible — unlike HH-RLHF, this path must filter.
    decontam_index = _build_dpo_decontam_index(dpo.decontam_datasets)

    n_train = 0
    n_dev = 0
    n_dropped_invalid = 0
    n_dropped_decontam = 0
    target_total = sum(s.target_pairs for s in dpo.sources)
    train_target = max(0, target_total - dpo.dev_pairs)
    per_source_counts: dict[str, int] = {}

    with train_path.open("w", encoding="utf-8") as f_train, dev_path.open("w", encoding="utf-8") as f_dev:
        for source in dpo.sources:
            if not source.loader:
                raise ValueError(
                    f"DPO source {source.name!r} requires a `loader` when "
                    f"preference_format=mc_letter (e.g. 'piqa', 'sciq', 'ai2_arc')."
                )
            log.info("Loading DPO MC source: %s (%s)", source.name, source.path)

            # Build a SFTSourceConfig-shaped object so the SFT normalizer
            # dispatch works.  We only need the fields the normalizers read.
            sft_like = SFTSourceConfig(
                name=source.name,
                loader=source.loader,
                path=source.path,
                subset=source.subset,
                split=source.split,
                target_examples=source.target_pairs,
                language=source.language,
                rationale="",
            )

            ds = load_dataset(
                source.path,
                source.subset,
                split=source.split,
                cache_dir=str(dpo.cache_dir) if dpo.cache_dir else None,
                token=hf_token,
            )

            count_for_source = 0
            target_for_source = source.target_pairs

            for raw in ds:
                if count_for_source >= target_for_source:
                    break
                if n_train >= train_target and n_dev >= dpo.dev_pairs:
                    break
                normalized = _normalize_source_record(raw, source_cfg=sft_like)
                if normalized is None:
                    n_dropped_invalid += 1
                    continue
                messages, _meta = normalized
                if (
                    len(messages) != 2
                    or messages[0].get("role") != "user"
                    or messages[1].get("role") != "assistant"
                ):
                    n_dropped_invalid += 1
                    continue
                user_prompt = str(messages[0].get("content") or "")
                correct_letter = str(messages[1].get("content") or "").strip().upper()
                if len(correct_letter) != 1 or not correct_letter.isalpha():
                    n_dropped_invalid += 1
                    continue

                # Eval-split decontamination: drop pairs whose prompt overlaps
                # an eval-split entry. Streaming pipeline -> check inline rather
                # than buffering the full pair list. The same logic is exposed
                # via `_filter_decontaminated` for tests / batch use.
                if decontam_index.contains(user_prompt):
                    n_dropped_decontam += 1
                    continue

                # Discover candidate letters by scanning the rendered option lines.
                # SFT MC normalizers use two prompt formats:
                #   - `_format_mc_prompt`  -> "A) ..." / "B) ..." (sciq, cqa, mmlu, race, boolq, piqa, wsc273)
                #   - `_normalize_ai2_arc_record` -> "A. ..." / "B. ..." (ARC easy/challenge)
                candidate_letters: list[str] = []
                for letter in "ABCDEFGHIJ":
                    paren = f"{letter}) "
                    dot = f"{letter}. "
                    if (
                        f"\n{paren}" in user_prompt
                        or user_prompt.startswith(paren)
                        or f"\n{dot}" in user_prompt
                        or user_prompt.startswith(dot)
                    ):
                        candidate_letters.append(letter)
                if correct_letter not in candidate_letters:
                    n_dropped_invalid += 1
                    continue

                packed = _build_letter_dpo_pair(
                    user_prompt=user_prompt,
                    correct_letter=correct_letter,
                    candidate_letters=candidate_letters,
                    tokenizer=tokenizer,
                    system_prompt=dpo.system_prompt,
                    max_seq_length=dpo.max_seq_length,
                    rng=rng,
                )
                if packed is None:
                    n_dropped_invalid += 1
                    continue

                # Fill dev first so smoke configs (small totals) still get a
                # nonzero dev split. After dev_pairs is met, all remaining
                # records flow to train until train_target is reached.
                if n_dev < dpo.dev_pairs:
                    f_dev.write(json.dumps(packed) + "\n")
                    n_dev += 1
                else:
                    f_train.write(json.dumps(packed) + "\n")
                    n_train += 1
                count_for_source += 1

            per_source_counts[source.name] = count_for_source
            log.info("Source %s: built %d letter pairs", source.name, count_for_source)
            if n_dev >= dpo.dev_pairs and n_train >= train_target:
                break

    manifest = {
        "preference_format": "mc_letter",
        "n_train": n_train,
        "n_dev": n_dev,
        "n_dropped_invalid": n_dropped_invalid,
        "n_dropped_decontam": n_dropped_decontam,
        "per_source": per_source_counts,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_kept = n_train + n_dev
    log.info(
        "DPO decontam: kept %d / %d pairs (%d dropped)",
        total_kept, total_kept + n_dropped_decontam, n_dropped_decontam,
    )
    log.info("DPO mc_letter prepare finished: %s", manifest)
