"""Prefetch and validate public SFT datasets before preparation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from configs import ProjectConfig, SFTDecontamConfig, SFTSourceConfig
from src.scripts.download_data import (
    download_fasttext_langdetect,
    download_nlp26_eval,
    download_wikitext103_test,
)
from src.utils import build_tokenizer

from .prepare import (
    OptionalLanguageFilter,
    PreparedExample,
    _collect_candidates_for_source,
    _cleanup_cache_dir,
    _extract_hf_prompt,
    _iter_local_disk_texts,
    _normalize_source_record,
    get_decontam_snapshot_path,
    get_source_snapshot_path,
)


log = logging.getLogger("parrotllm.posttraining")


def _load_hf_split(
    path: str,
    subset: str | None,
    split: str,
    *,
    cache_dir: Path | None,
    hf_token: str | None,
):
    kwargs: dict[str, Any] = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if hf_token is not None:
        kwargs["token"] = hf_token
    return load_dataset(path, subset, **kwargs)


def _dataset_num_rows(dataset) -> int | None:
    value = getattr(dataset, "num_rows", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _matches_suffix(path: Path, expected_suffix: str) -> bool:
    actual = str(path).replace("\\", "/").rstrip("/")
    suffix = expected_suffix.replace("\\", "/").rstrip("/")
    return actual.endswith(suffix)


def _bootstrap_public_local_assets(project_config: ProjectConfig) -> dict[str, str]:
    sft_cfg = project_config.sft
    assert sft_cfg is not None

    bootstrapped: dict[str, str] = {}
    if any(source.language for source in sft_cfg.sources):
        download_fasttext_langdetect()
        bootstrapped["fasttext_langdetect"] = "ok"

    for dataset_cfg in sft_cfg.decontam_datasets:
        if dataset_cfg.loader != "local_disk":
            continue
        local_path = Path(dataset_cfg.path)
        if _matches_suffix(local_path, "data/wikitext-103-test"):
            download_wikitext103_test()
            bootstrapped[dataset_cfg.name] = "downloaded_via_download_data.py"
        elif _matches_suffix(local_path, "data/owt-eval/NLP26/NLP26_OWT_eval/test"):
            download_nlp26_eval()
            bootstrapped[dataset_cfg.name] = "downloaded_via_download_data.py"
    return bootstrapped


def _validate_source_preview(
    source_cfg: SFTSourceConfig,
    *,
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
    lang_filter: OptionalLanguageFilter,
    snapshot_path: Path,
    cache_dir: Path | None,
    hf_token: str | None,
) -> dict[str, Any]:
    if source_cfg.loader == "local_jsonl":
        raise ValueError(
            f"SFT source '{source_cfg.name}' uses loader=local_jsonl. "
            "The sft-download stage only handles public datasets."
        )

    dataset = load_from_disk(str(snapshot_path))
    preview_cfg = source_cfg.model_copy(
        update={
            "target_examples": min(8, source_cfg.target_examples),
            "candidate_multiplier": min(2, source_cfg.candidate_multiplier),
        }
    )
    preview_candidates: list[PreparedExample] = _collect_candidates_for_source(
        preview_cfg,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_seq_length=max_seq_length,
        lang_filter=lang_filter,
        snapshot_path=snapshot_path,
    )
    raw_normalized_examples = 0
    validation_mode = "strict_preview"
    if not preview_candidates:
        # Download validation should verify that the public source is reachable and schema-compatible.
        # If the strict SFT filters are temporarily too aggressive, fall back to checking that at least
        # some rows from the downloaded dataset can still be normalized into messages.
        for row in dataset:
            normalized = _normalize_source_record(row, source_cfg=source_cfg)
            if normalized is None:
                continue
            messages, _metadata = normalized
            if messages:
                raw_normalized_examples += 1
            if raw_normalized_examples >= 3:
                break
        if raw_normalized_examples <= 0:
            raise RuntimeError(
                f"SFT source '{source_cfg.name}' downloaded successfully but yielded zero "
                "preview examples after normalization/filtering and zero raw normalized examples. "
                "This usually means the configured source filter no longer matches the dataset schema."
            )
        validation_mode = "fallback_raw_schema"
        log.warning(
            "SFT source %s passed only fallback validation: %d raw normalized examples, "
            "but zero strict preview candidates. sft-prepare may still underfill this slice.",
            source_cfg.name,
            raw_normalized_examples,
        )
    return {
        "name": source_cfg.name,
        "loader": source_cfg.loader,
        "path": source_cfg.path,
        "subset": source_cfg.subset,
        "split": source_cfg.split,
        "snapshot_path": str(snapshot_path),
        "num_rows": _dataset_num_rows(dataset),
        "preview_candidates": len(preview_candidates),
        "raw_normalized_examples": raw_normalized_examples,
        "validation_mode": validation_mode,
        "target_examples": source_cfg.target_examples,
    }


def _validate_local_jsonl_source(
    source_cfg: SFTSourceConfig,
    *,
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
    lang_filter: OptionalLanguageFilter,
) -> dict[str, Any]:
    if source_cfg.loader != "local_jsonl":
        raise ValueError(f"Expected local_jsonl source, got {source_cfg.loader!r}.")

    local_path = Path(source_cfg.path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local SFT source '{source_cfg.name}' is missing at {local_path}."
        )

    preview_cfg = source_cfg.model_copy(
        update={
            "target_examples": min(8, source_cfg.target_examples),
            "candidate_multiplier": min(2, source_cfg.candidate_multiplier),
        }
    )
    preview_candidates: list[PreparedExample] = _collect_candidates_for_source(
        preview_cfg,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_seq_length=max_seq_length,
        lang_filter=lang_filter,
    )
    if not preview_candidates:
        raise RuntimeError(
            f"Local SFT source '{source_cfg.name}' at {local_path} yielded zero "
            "preview examples after normalization/filtering."
        )

    return {
        "name": source_cfg.name,
        "loader": source_cfg.loader,
        "path": str(local_path),
        "snapshot_path": None,
        "num_rows": None,
        "preview_candidates": len(preview_candidates),
        "raw_normalized_examples": len(preview_candidates),
        "validation_mode": "local_jsonl_preview",
        "target_examples": source_cfg.target_examples,
    }


def _validate_local_decontam(cfg: SFTDecontamConfig) -> dict[str, Any]:
    local_path = Path(cfg.path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local decontamination asset '{cfg.name}' is missing at {local_path}."
        )

    sample_count = 0
    for text in _iter_local_disk_texts(local_path, cfg.field):
        if str(text).strip():
            sample_count += 1
        if sample_count >= 3:
            break

    if sample_count <= 0:
        raise RuntimeError(
            f"Local decontamination asset '{cfg.name}' at {local_path} contains no readable text."
        )

    return {
        "name": cfg.name,
        "loader": cfg.loader,
        "path": str(local_path),
        "sample_texts": sample_count,
    }


def _validate_hf_decontam(
    cfg: SFTDecontamConfig,
    *,
    snapshot_path: Path,
) -> dict[str, Any]:
    dataset = load_from_disk(str(snapshot_path))
    sample_count = 0
    for row in dataset:
        text = _extract_hf_prompt(row, cfg.name, cfg.field)
        if text and str(text).strip():
            sample_count += 1
        if sample_count >= 3:
            break
    if sample_count <= 0:
        raise RuntimeError(
            f"HF decontamination dataset '{cfg.name}' yielded no readable prompts."
        )
    return {
        "name": cfg.name,
        "loader": cfg.loader,
        "path": cfg.path,
        "subset": cfg.subset,
        "split": cfg.split,
        "snapshot_path": str(snapshot_path),
        "num_rows": _dataset_num_rows(dataset),
        "sample_prompts": sample_count,
    }


def _materialize_hf_dataset(dataset, snapshot_path: Path) -> Path:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = snapshot_path.with_name(f"{snapshot_path.name}.tmp")
    if staging_path.exists():
        shutil.rmtree(staging_path, ignore_errors=True)
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path, ignore_errors=True)
    dataset.save_to_disk(str(staging_path))
    staging_path.replace(snapshot_path)
    return snapshot_path


def run_download_sft(
    project_config: ProjectConfig,
    *,
    hf_token: str | None = None,
) -> dict[str, Any]:
    sft_cfg = project_config.sft
    if sft_cfg is None:
        raise ValueError("SFT configuration missing; cannot download posttraining data.")

    prepared_dir = Path(sft_cfg.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(sft_cfg.cache_dir) if sft_cfg.cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(sft_cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    bootstrapped_assets = _bootstrap_public_local_assets(project_config)
    lang_filter = OptionalLanguageFilter()

    snapshot_cache: dict[tuple[str, str | None, str], Path] = {}
    source_results = []
    for source_cfg in sft_cfg.sources:
        if source_cfg.loader == "local_jsonl":
            source_results.append(
                _validate_local_jsonl_source(
                    source_cfg,
                    tokenizer=tokenizer,
                    system_prompt=sft_cfg.system_prompt,
                    max_seq_length=sft_cfg.max_seq_length,
                    lang_filter=lang_filter,
                )
            )
            continue
        dataset_key = (source_cfg.path, source_cfg.subset, source_cfg.split)
        snapshot_path = snapshot_cache.get(dataset_key)
        if snapshot_path is None:
            snapshot_path = get_source_snapshot_path(raw_dir, source_cfg)
            assert snapshot_path is not None
            if snapshot_path.exists():
                log.info("Reusing saved SFT source snapshot: %s", snapshot_path)
            else:
                dataset = _load_hf_split(
                    source_cfg.path,
                    source_cfg.subset,
                    source_cfg.split,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                )
                _materialize_hf_dataset(dataset, snapshot_path)
            snapshot_cache[dataset_key] = snapshot_path
        source_results.append(
            _validate_source_preview(
                source_cfg,
                tokenizer=tokenizer,
                system_prompt=sft_cfg.system_prompt,
                max_seq_length=sft_cfg.max_seq_length,
                lang_filter=lang_filter,
                snapshot_path=snapshot_path,
                cache_dir=cache_dir,
                hf_token=hf_token,
            )
        )

    decontam_results = []
    decontam_snapshot_cache: dict[tuple[str, str | None, str], Path] = {}
    for cfg in sft_cfg.decontam_datasets:
        if not cfg.enabled:
            continue
        if cfg.loader == "local_disk":
            decontam_results.append(_validate_local_decontam(cfg))
        else:
            dataset_key = (cfg.path, cfg.subset, cfg.split)
            snapshot_path = decontam_snapshot_cache.get(dataset_key)
            if snapshot_path is None:
                snapshot_path = get_decontam_snapshot_path(raw_dir, cfg)
                assert snapshot_path is not None
                if snapshot_path.exists():
                    log.info("Reusing saved SFT decontam snapshot: %s", snapshot_path)
                else:
                    dataset = _load_hf_split(
                        cfg.path,
                        cfg.subset,
                        cfg.split,
                        cache_dir=cache_dir,
                        hf_token=hf_token,
                    )
                    _materialize_hf_dataset(dataset, snapshot_path)
                decontam_snapshot_cache[dataset_key] = snapshot_path
            decontam_results.append(
                _validate_hf_decontam(
                    cfg,
                    snapshot_path=snapshot_path,
                )
            )

    cache_cleaned = _cleanup_cache_dir(cache_dir, protected_paths=[prepared_dir, raw_dir])
    manifest = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "cache_cleaned": cache_cleaned,
        "raw_dir": str(raw_dir),
        "prepared_dir": str(prepared_dir),
        "bootstrapped_assets": bootstrapped_assets,
        "sources": source_results,
        "decontam_datasets": decontam_results,
        "total_target_examples": sum(source_cfg.target_examples for source_cfg in sft_cfg.sources),
    }
    manifest_path = prepared_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("SFT download manifest written to %s", manifest_path)
    return manifest


__all__ = ["run_download_sft"]
