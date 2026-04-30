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
    prompt_render = render_conversation(prompt_messages)
    # Append the assistant header so the response starts at a clean position.
    prompt_text = prompt_render.text + "\n\n### Response:\n"

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
    """Phase 1 stub: returns empty set.

    Real decontam against the SFT eval corpora (HellaSwag/WinoGrande/OBQA/LAMBADA)
    would use src.posttraining.prepare._build_decontam_index + PromptContaminationIndex.
    Skipped for Phase 1 because HH-RLHF prompts rarely overlap with cloze benchmarks
    in practice. Re-enable here if Phase 1 acceptance turns up suspected contamination.
    """
    specs_list = list(decontam_specs)
    if specs_list:
        log.info(
            "DPO decontam stub active — %d decontam_datasets entries ignored. "
            "Wire to PromptContaminationIndex if benchmarks regress unexpectedly.",
            len(specs_list),
        )
    return set()


def run_prepare_dpo(project_config: ProjectConfig, *, seed: int, hf_token: str | None = None) -> None:
    """Pipeline entry point: download HH-RLHF, format, decontam, write JSONL."""
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
                if n_dev >= dpo.dev_pairs and n_train >= train_target:
                    break

    manifest = {
        "n_train": n_train,
        "n_dev": n_dev,
        "n_dropped_decontam": n_dropped_decontam,
        "n_dropped_invalid": n_dropped_invalid,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("DPO prepare finished: %s", manifest)
