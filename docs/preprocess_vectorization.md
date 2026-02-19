# Vectorizing `src/data/preprocess.py`

`src/data/preprocess.py` implements the production preprocessing pipeline: build contamination indexes, sanitize and filter documents, tokenize, and write train/val binaries. Most steps iterate through every document inside Python, which keeps the code readable but leaves a lot of CPU parallelism and vectorized primitives unused. Below is a walkthrough of the heavy sections and specific ways to speed them up.

## 1. Building the test decontamination index (`build_test_decontamination_index`)

**Current behavior**
- Loads each eval split with `load_from_disk` and iterates row by row.
- For every doc it normalizes text, extracts 13-gram character sets, hashes content, and (optionally) computes MinHash shingles/signatures.
- All string manipulation happens in pure Python; `tqdm` only shows progress.

**Opportunities**
1. Use `dataset.map(..., batched=True, num_proc=os.cpu_count())` so normalization, hashing, and shingle generation run in parallel workers. Each batch can return dictionaries of `ngrams`, `fingerprints`, `shingle_hashes`, etc., which you reduce afterward.
2. Replace Python n-gram sets with hashed integer representations (e.g., rolling hash or `np.lib.stride_tricks.sliding_window_view` over UTF-8 bytes). A `np.unique` on those arrays is far faster than Python `set.update` per document.
3. Persist intermediate artifacts (e.g., fingerprint arrays, MinHash signatures) to disk so re-running `build_test_decontamination_index` skips recomputation.

## 2. Main document loop (`run_preprocess` → `preprocess_document`)

**Current behavior**
- Sequentially loops through every document in the dataset (`for doc in tqdm(ds)`).
- For each doc it runs `sanitize_document_text`, `detect_language`, normalization + fingerprinting, MinHash overlap, contamination n-gram check, and finally `tokenizer.encode`.
- Keeps a Python list `all_tokens` and repeatedly `extend`s it, then converts to `np.array` once at the end.

**Opportunities**
1. Convert the entire pipeline to a single batched `Dataset.map` that returns token lists and status labels. Set `num_proc=os.cpu_count()` and tune `batch_size` (e.g., 64–256) so the dataset library parallelizes the expensive sections automatically.
2. FastText supports batch prediction: call `lang_model.predict(batch_first_lines)` once per batch to avoid Python function overhead.
3. Use the tokenizer’s batch interface (`tokenizer(texts, add_special_tokens=False, return_attention_mask=False)`) instead of `encode` inside a loop. This keeps tokenization in Rust and yields ready-to-use `input_ids` arrays for the subsequent filters.
4. Avoid repeatedly generating doc-level n-gram sets. Pre-hash test n-grams (integers) and compare against doc hashes with `np.isin` or `np.intersect1d` inside the batched map. That shrinks both CPU time and Python object churn.
5. Replace the ever-growing `all_tokens` Python list with chunked NumPy arrays or a memory-mapped writer. For example, collect `np.array(batch_tokens, dtype=np.uint16)` per batch and append to a list of arrays, then `np.concatenate` once. For very large corpora, write batches straight into `np.memmap` files to avoid duplicating data in RAM.
6. Push the train/val split logic into streaming writes: once you know the total token count (via a dry run or metadata), preallocate two memmaps and copy slices as you go instead of materializing the entire sequence.

## 3. Sanitization helpers (HTML stripping, boilerplate removal, code detection)

**Current behavior**
- Operate on one document at a time with Python loops, regex, and string slicing.

**Opportunities**
1. Move them into batched dataset operations so multiple documents are cleaned per worker. Regex functions like `re.sub` already operate on Python strings but eliminating the outer for-loop saves interpreter overhead.
2. Cache compiled regex objects at module load (already done) but ensure they are reused across processes by defining them at the top level, which aligns with `num_proc` usage.
3. For code detection, compute features (punctuation ratios, keyword hits) with NumPy vectorization or pandas string methods if you convert a batch into a Series—significantly faster than Python counting inside nested loops.

## 4. MinHash and shingle generation

**Current behavior**
- `compute_shingle_hashes` splits text into tokens, then loops to create each shingle and hash it with `hashlib.blake2b`.

**Opportunities**
1. Use a rolling-hash approach to compute shingle hashes in O(n) without slicing/joining strings per shingle.
2. Convert the token list into a NumPy array of integer IDs (e.g., hashed tokens) and use `np.lib.stride_tricks.sliding_window_view` to obtain all shingles, then hash the slices with vectorized operations.
3. Batch shingle hashing across documents to better utilize CPU caches.

## 5. Stats and logging

**Current behavior**
- Filter statistics are gathered via a `Counter` updated per document.

**Opportunities**
- When switching to batched processing, accumulate stats per batch and reduce at the end (`Counter.update`). This reduces lock contention and makes it easy to emit intermediate metrics (e.g., each worker can report counts via shared queues if needed).

## Putting it together

1. Refactor `preprocess_document` into a function that accepts entire batches of texts and returns filtered token arrays plus metadata. Hugging Face `Dataset.map` can orchestrate batching/multiprocessing automatically.
2. Build decontamination indexes once per run using the same batched approach; store them in `data/cache/…` so notebooks/scripts can reuse them without recomputing.
3. Switch to chunked / memory-mapped token output to minimize RAM usage and avoid the large `all_tokens` list.
4. After the refactor, expose lightweight helpers (e.g., `prepare_batch(batch_texts)` and `build_test_index(data_dir)`) so notebooks and CLI tools share exactly the same vectorized logic.

Adopting these steps keeps behavior identical but lets the preprocessing stage saturate CPU cores and compiled tokenizer/fastText code, which should cut runtime by an order of magnitude on multi-core machines.
