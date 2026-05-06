"""Helpers for keeping Hugging Face dataset caches small.

Post-training pulls several HF datasets only to materialize them into local
Python/tokenized rows. On small Mac disks, leaving those Arrow and Hub dataset
caches behind can consume many GB across repeated SFT/DPO experiments.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path


log = logging.getLogger("parrotllm.hf_cache")


def _default_datasets_cache() -> Path | None:
    try:
        from datasets import config as datasets_config  # type: ignore[import]
    except Exception:
        return None
    cache = getattr(datasets_config, "HF_DATASETS_CACHE", None)
    return Path(cache).expanduser() if cache else None


def _default_hub_cache() -> Path | None:
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return Path(cache).expanduser()
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore[import]
    except Exception:
        return None
    return Path(HF_HUB_CACHE).expanduser()


def cleanup_hf_dataset_cache(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    include_hub_dataset_repos: bool = True,
) -> list[Path]:
    """Delete local HF dataset cache files and return removed paths.

    ``cache_dir`` should match the value passed to ``datasets.load_dataset``.
    When omitted, the function removes the default ``datasets`` cache and the
    dataset snapshots under the Hugging Face Hub cache (``datasets--*``).

    The function intentionally does not delete model/tokenizer Hub repos.
    """
    removed: list[Path] = []
    targets: list[Path] = []

    if cache_dir:
        targets.append(Path(cache_dir).expanduser())
    else:
        datasets_cache = _default_datasets_cache()
        if datasets_cache is not None:
            targets.append(datasets_cache)

    if include_hub_dataset_repos:
        hub_cache = _default_hub_cache()
        if hub_cache is not None and hub_cache.exists():
            targets.extend(sorted(hub_cache.glob("datasets--*")))

    for target in targets:
        if not target.exists():
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError as exc:
            log.warning("HF cache cleanup: could not remove %s: %s", target, exc)
            continue
        removed.append(target)

    if removed:
        log.info(
            "HF cache cleanup: removed %d dataset cache path(s): %s",
            len(removed),
            ", ".join(str(p) for p in removed[:5])
            + (" ..." if len(removed) > 5 else ""),
        )
    else:
        log.info("HF cache cleanup: no dataset cache paths found to remove.")

    return removed
