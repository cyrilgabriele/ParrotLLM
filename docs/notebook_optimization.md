# Speeding Up `01_data_preprocessing.ipynb`

Slow cells in `src/notebooks/01_data_preprocessing.ipynb` mostly come from repeatedly looping over the dataset inside Python and re-doing heavy operations. The notes below capture the main hotspots and how to vectorize them.

## Token length statistics

Current code loops over each document and calls `tokenizer.encode` sequentially. Use the tokenizer's batch API instead:

```python
enc = tokenizer(ds_small["text"], add_special_tokens=False, return_length=True)
doc_lengths = np.array(enc["length"])
```

This runs the Rust tokenizer once over the whole batch, gives you the lengths, and lets you reuse `enc["input_ids"]` later instead of re-tokenizing.

## Language detection + preprocessing

Both the lang distribution cell and the `preprocess_document` loop iterate over `ds_small`. Convert them into a single batched `Dataset.map`/`filter` pipeline:

- fastText's `predict` accepts a list of strings, so detect languages for an entire batch in one call instead of per doc.
- Run the contamination check and tokenization inside the same batched function, then call `dataset.filter` to keep only rows passing the criteria.
- Set `num_proc=os.cpu_count()` and a reasonable `batch_size` so the dataset library parallelizes across CPU cores.

This way every document is read once and all heavy operations run in parallel workers.

## Duplicate detection

The current fingerprinting logic normalizes strings in a Python loop. Add a `fingerprint` column via `ds_small.map(..., batched=True)` where you lowercase, collapse whitespace, and hash the batch using NumPy/pandas-style vector ops. Afterward group by fingerprint (`Dataset.to_pandas().groupby`) to list duplicates without scanning everything in pure Python.

## Test-set contamination

`extract_ngrams` rebuilds character-level sets per call. Precompute the n-gram hashes for the test sets using `np.lib.stride_tricks.sliding_window_view` (or another vectorized method) and store them in a hashed set. Inside the batched preprocessing function compare against that hashed set so each doc just runs a vectorized overlap check rather than Python set intersections.

## General tips

- Reuse tokenizer outputs instead of re-running `.encode` in multiple cells.
- Persist intermediate arrays to disk (e.g., token IDs, language labels) and reload them in later cells to avoid repeating work after a kernel restart.
- Keep the notebook focused on inspection; move the final `Dataset.map` pipeline into a script so the heavy lifting runs once from the CLI.
