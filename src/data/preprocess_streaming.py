"""Time-bounded streaming preprocessing pipeline.

Instead of pre-downloading a fixed document subset and then processing it
sequentially, this pipeline streams documents from HuggingFace, applies
all filter phases to each micro-batch as it arrives, and stops accepting
new documents when the wall-clock time budget expires.  Deduplication,
tokenization, and binary output happen once at the end on the accumulated
survivors.

Usage:
    uv run python main.py --stage preprocess-streaming \
        --config configs/preprocessing/preprocess_var_c_streaming.yaml

The pipeline reuses all filter functions from ``src.data.preprocess`` so
that the same filtering logic applies regardless of which pipeline is used.
"""

from __future__ import annotations

import itertools
import logging
import os
import random as _random
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/parrotllm_hf_cache")
from datasets import Dataset, disable_caching

from configs.preprocessing.preprocessConfig import StreamingPreprocessConfig
from src.data.preprocess import (
    DecontaminationIndex,
    FINGERPRINT_LOWERCASE,
    MinHasher,
    TOPIC_LABEL_MAP,
    UnionFind,
    _shingle_hashes,
    build_test_decontamination_index,
    decontaminate_batch,
    ellipsis_filter_batch,
    heuristic_code_filter_batch,
    heuristic_quality_filter_batch,
    sanitize_batch,
    detect_language_batch,
    _get_topic_pipeline,
)
from src.utils import build_tokenizer

log = logging.getLogger("parrotllm.preprocess_streaming")


# ── Micro-batch filter ──────────────────────────────────────────────────────


def _filter_micro_batch(
    texts: list[str],
    *,
    args: StreamingPreprocessConfig,
    test_index: DecontaminationIndex,
    lang_model_path: str,
    topic_pipeline,
) -> list[str]:
    """Run phases 1-5 + 6.1 on a list of raw texts, return survivors."""

    # Phase 1: Decontamination
    if not args.skip_decontam:
        statuses = decontaminate_batch(texts, test_index)["decontam_status"]
        texts = [t for t, s in zip(texts, statuses) if s == "kept"]
        if not texts:
            return []

    # Phase 2: Sanitization
    result = sanitize_batch(texts)
    texts = [t for t, s in zip(result["text"], result["sanitize_status"]) if s == "kept" and t]
    if not texts:
        return []

    # Phase 3: Language filter
    result = detect_language_batch(texts, lang_model_path)
    texts = [
        t for t, lang, conf in zip(texts, result["lang"], result["lang_conf"])
        if lang == args.lang and conf >= args.language_confidence_threshold
    ]
    if not texts:
        return []

    # Phase 3.5: Topic filter
    topic_classes = args.topic_classes
    if topic_classes and not args.skip_topic_filter and topic_pipeline is not None:
        truncated = [t[:args.topic_text_truncation] for t in texts]
        results = topic_pipeline(truncated, truncation=True, batch_size=args.topic_batch_size)
        labels = [TOPIC_LABEL_MAP.get(r["label"], r["label"]) for r in results]
        texts = [t for t, lbl in zip(texts, labels) if lbl in topic_classes]
        if not texts:
            return []

    # Phase 4: Code / artifact filter
    filter_mode = args.filter_mode
    if filter_mode != "none" and not args.skip_code_filter:
        if filter_mode == "heuristic":
            statuses = heuristic_code_filter_batch(texts)["code_filter_status"]
            texts = [t for t, s in zip(texts, statuses) if s == "kept"]
            if not texts:
                return []

    # Phase 5: Quality / coherence filter
    if filter_mode != "none" and not args.skip_quality_filter:
        if filter_mode == "heuristic":
            statuses = heuristic_quality_filter_batch(texts)["quality_filter_status"]
            texts = [t for t, s in zip(texts, statuses) if s == "kept"]
            if not texts:
                return []

    # Phase 6.1: Ellipsis filter
    if not args.skip_ellipsis_filter:
        statuses = ellipsis_filter_batch(texts)["ellipsis_filter_status"]
        texts = [t for t, s in zip(texts, statuses) if s == "kept"]

    return texts


# ── Dedup on accumulated survivors ──────────────────────────────────────────


def _deduplicate_texts(
    texts: list[str],
    num_perm: int,
    bands: int,
    rows: int,
    shingle_size: int,
) -> list[str]:
    """MinHash-LSH dedup over a plain list of strings. Returns deduplicated list."""
    n = len(texts)
    if n < 2:
        return texts

    hasher = MinHasher(num_perm=num_perm, seed=42)

    log.info("  Computing MinHash signatures for %s docs...", f"{n:,}")
    sigs = np.array(
        [hasher.signature(_shingle_hashes(t, shingle_size)) for t in texts],
        dtype=np.int64,
    )

    uf = UnionFind()
    for band_idx in range(bands):
        start = band_idx * rows
        band_sigs = sigs[:, start : start + rows]
        buckets: dict[bytes, list[int]] = {}
        for doc_idx in range(n):
            key = band_sigs[doc_idx].tobytes()
            buckets.setdefault(key, []).append(doc_idx)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            first = bucket[0]
            for other in bucket[1:]:
                uf.union(first, other)

    to_remove: set[int] = set()
    root_seen: set[int] = set()
    for doc_idx in range(n):
        if doc_idx not in uf.parent:
            continue
        root = uf.find(doc_idx)
        if root in root_seen:
            to_remove.add(doc_idx)
        else:
            root_seen.add(root)

    kept = [t for i, t in enumerate(texts) if i not in to_remove]
    log.info("  Dedup: kept %s / %s (removed %s)", f"{len(kept):,}", f"{n:,}", f"{len(to_remove):,}")
    return kept


# ── Main entry point ────────────────────────────────────────────────────────


def run_preprocess_streaming(args: StreamingPreprocessConfig) -> None:
    """Run the streaming preprocessing pipeline with a wall-clock time budget."""
    disable_caching()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    max_time = args.max_time_seconds

    log.info("=" * 60)
    log.info("Streaming preprocessing pipeline")
    log.info("  Time budget: %s seconds (%.1f hours)", f"{max_time:,}", max_time / 3600)
    log.info("  Micro-batch size: %s docs", f"{args.stream_batch_size:,}")
    log.info("=" * 60)

    # ── Load models / indexes upfront ────────────────────────────────────
    tokenizer = build_tokenizer(
        add_prefix_space=False,
        padding_side="right",
        tokenizer_name=args.tokenizer_name,
    )
    log.info("Tokenizer loaded (vocab_size=%s)", f"{len(tokenizer):,}")

    lang_model_path = str(data_dir / "lid.176.ftz")
    if not Path(lang_model_path).exists():
        raise FileNotFoundError(
            f"Language model not found at {lang_model_path}. "
            "Run: uv run python src/scripts/download_data.py lang-model"
        )

    if args.skip_decontam:
        test_index = DecontaminationIndex(content_hashes=set())
    else:
        test_index = build_test_decontamination_index(data_dir, lowercase=FINGERPRINT_LOWERCASE)

    topic_pipeline = None
    if args.topic_classes and not args.skip_topic_filter:
        topic_pipeline = _get_topic_pipeline()

    # ── Stream + filter ──────────────────────────────────────────────────
    from datasets import load_dataset

    log.info("Opening HF stream (seed=%d, buffer=%s)...", seed, f"{args.shuffle_buffer_size:,}")
    ds_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds_stream = ds_stream.shuffle(seed=seed, buffer_size=args.shuffle_buffer_size)

    survivors: list[str] = []
    total_streamed = 0
    total_kept = 0
    batch_buffer: list[str] = []

    t_start = time.monotonic()

    for doc in ds_stream:
        batch_buffer.append(doc["text"])
        total_streamed += 1

        if len(batch_buffer) >= args.stream_batch_size:
            kept = _filter_micro_batch(
                batch_buffer,
                args=args,
                test_index=test_index,
                lang_model_path=lang_model_path,
                topic_pipeline=topic_pipeline,
            )
            survivors.extend(kept)
            total_kept += len(kept)
            batch_buffer = []

            elapsed = time.monotonic() - t_start
            if total_streamed % (args.stream_batch_size * 10) == 0:
                log.info(
                    "  [%.0fs / %ss] streamed %s, kept %s (%.1f%%)",
                    elapsed, max_time,
                    f"{total_streamed:,}", f"{total_kept:,}",
                    100.0 * total_kept / max(total_streamed, 1),
                )

            if elapsed >= max_time:
                log.info("Time budget reached (%.0fs). Stopping stream.", elapsed)
                break

    # Flush remaining buffer
    if batch_buffer:
        kept = _filter_micro_batch(
            batch_buffer,
            args=args,
            test_index=test_index,
            lang_model_path=lang_model_path,
            topic_pipeline=topic_pipeline,
        )
        survivors.extend(kept)
        total_kept += len(kept)

    elapsed_stream = time.monotonic() - t_start
    log.info(
        "Streaming complete: %s docs streamed, %s survivors (%.1f%%) in %.0fs",
        f"{total_streamed:,}", f"{len(survivors):,}",
        100.0 * len(survivors) / max(total_streamed, 1),
        elapsed_stream,
    )

    if not survivors:
        log.warning("No documents survived filtering. Nothing to write.")
        return

    # ── Phase 6: Dedup ───────────────────────────────────────────────────
    if not args.skip_dedup:
        log.info("Phase 6: Fuzzy deduplication on %s survivors...", f"{len(survivors):,}")
        survivors = _deduplicate_texts(
            survivors,
            num_perm=args.dedup_num_perm,
            bands=args.dedup_bands,
            rows=args.dedup_rows,
            shingle_size=args.dedup_shingle_size,
        )

    # ── Phase 3.5 (post-hoc): Topic distribution resampling ─────────────
    # The streaming loop already filtered by topic class. If a target
    # distribution was requested, resample the survivors to match it.
    if (
        args.topic_classes
        and not args.skip_topic_filter
        and args.topic_distribution
        and topic_pipeline is not None
    ):
        log.info("Resampling survivors to match topic distribution...")
        truncated = [t[:args.topic_text_truncation] for t in survivors]
        # Classify in batches to avoid OOM on large survivor sets
        labels: list[str] = []
        bs = args.topic_batch_size
        for i in range(0, len(truncated), bs):
            results = topic_pipeline(truncated[i : i + bs], truncation=True, batch_size=bs)
            labels.extend(TOPIC_LABEL_MAP.get(r["label"], r["label"]) for r in results)

        rng = _random.Random(seed)
        weight_sum = sum(args.topic_distribution.values())
        class_indices: dict[str, list[int]] = {cls: [] for cls in args.topic_classes}
        for idx, lbl in enumerate(labels):
            if lbl in class_indices:
                class_indices[lbl].append(idx)

        total_available = sum(len(v) for v in class_indices.values())
        final_indices: list[int] = []
        for cls in args.topic_classes:
            available = class_indices.get(cls, [])
            if cls in args.topic_distribution:
                target_count = int(total_available * args.topic_distribution[cls] / weight_sum)
                if len(available) < target_count:
                    log.warning("'%s': only %s docs, target was %s — using all.", cls, f"{len(available):,}", f"{target_count:,}")
                    final_indices.extend(available)
                else:
                    final_indices.extend(rng.sample(available, target_count))
            else:
                final_indices.extend(available)

        final_indices.sort()
        survivors = [survivors[i] for i in final_indices]
        log.info("  After resampling: %s docs", f"{len(survivors):,}")

    # ── Phase 7: Tokenization ────────────────────────────────────────────
    log.info("Phase 7: Tokenizing %s documents...", f"{len(survivors):,}")
    _t = time.monotonic()

    all_ids: list[list[int]] = []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    tok_batch_size = 2048
    for i in range(0, len(survivors), tok_batch_size):
        batch = survivors[i : i + tok_batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)
        for ids in encoded["input_ids"]:
            if args.append_eos_token:
                ids.append(eos_id)
            if len(ids) >= args.minimum_tokens_per_doc:
                all_ids.append(ids)

    log.info(
        "  Tokenized: %s docs kept (%.0fs)",
        f"{len(all_ids):,}", time.monotonic() - _t,
    )

    if not all_ids:
        log.warning("No documents survived tokenization. Nothing to write.")
        return

    # ── Phase 8: Binary output ───────────────────────────────────────────
    log.info("Phase 8: Writing binary output...")
    n_tokens_list = [len(ids) for ids in all_ids]
    total_tokens = sum(n_tokens_list)

    token_array = np.fromiter(
        itertools.chain.from_iterable(all_ids),
        dtype=np.uint16,
        count=total_tokens,
    )
    log.info("  Total tokens: %s", f"{total_tokens:,}")

    n_val = max(1, int(len(token_array) * args.validation_split_ratio))
    val_tokens = token_array[:n_val]
    train_tokens = token_array[n_val:]

    train_windows = max(0, len(train_tokens) - args.context_length)
    val_windows = max(0, len(val_tokens) - args.context_length)
    log.info(
        "  train: %s tokens -> %s sliding-window starts (context_length=%d)",
        f"{len(train_tokens):,}", f"{train_windows:,}", args.context_length,
    )
    log.info(
        "  val:   %s tokens -> %s sliding-window starts",
        f"{len(val_tokens):,}", f"{val_windows:,}",
    )

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_tokens.tofile(str(train_path))
    val_tokens.tofile(str(val_path))

    log.info("train: %.1f MB -> %s", train_tokens.nbytes / 1e6, train_path)
    log.info("val:   %.1f MB -> %s", val_tokens.nbytes / 1e6, val_path)

    total_elapsed = time.monotonic() - t_start
    log.info(
        "Done. Total wall time: %.0fs (%.1f hours). "
        "Stream: %.0fs, post-processing: %.0fs.",
        total_elapsed, total_elapsed / 3600,
        elapsed_stream, total_elapsed - elapsed_stream,
    )

    # Clean up temporary HF cache
    _hf_cache = Path(os.environ.get("HF_DATASETS_CACHE", "/tmp/parrotllm_hf_cache"))
    if _hf_cache.exists() and str(_hf_cache).startswith("/tmp"):
        shutil.rmtree(_hf_cache, ignore_errors=True)
        log.info("Removed temporary HF cache at %s", _hf_cache)
