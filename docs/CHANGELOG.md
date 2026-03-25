# Changelog

Track what was changed, why it was changed, and any important notes.

## Entry Format

```markdown
### [YYYY-MM-DD] - [Contributor Name]

#### What
- List changes here

#### Why
- Explain reasoning

#### Remarks
- Optional notes, issues, or future work
```

---

## Unreleased

### [2026-03-25] - Christof Steiner

#### What
- Fixed 6 failing tests on main caused by config/test drift after architecture upgrade
- Added missing `lr_schedule`, `lr_decay_ratio`, `z_loss_coeff` fields to `configs/tune.yaml`
- Updated `test_parameter_count` to match actual architecture (Peri-LN, QK-Norm, SwiGLU 3-proj, RoPE)
- Fixed `test_deduplicate_keeps_longest` → `test_deduplicate_keeps_first` to match intentional keep-first dedup strategy
- Fixed `test_matches_sequential` in tokenization tests to account for EOS appending
- Renamed `TestDecontaminationIndex` → `DecontaminationIndex` to suppress pytest collection warning

#### Why
- Config schema evolved with the architecture upgrade but `tune.yaml` was not updated
- Parameter count test was stale from the original LayerNorm/GELU/learned-pos-emb baseline
- Dedup test expected keep-longest but code intentionally keeps lowest-index to avoid loading text column at scale
- Batch tokenizer appends EOS per document; sequential test path did not mirror this

#### Remarks


---
### [2026-03-25] - Cyril Gabriele

#### What
- Added torch.distributed-aware training: single run directory + logging only on rank 0, DDP wrapping, gradient accumulation with `no_sync`, and per-rank DistributedSamplers with `set_epoch`
- Wired VL06 recommendations into the default config (per-GPU batch 16, grad accumulation 2, `num_workers=4`, `pin_memory=true`) and propagated the loader knobs through the trainer/eval path
- Guarded evaluation/checkpointing/JSON logging to rank 0, added run-wide metric broadcasts, and ensured BF16/FP16 checkpoints unwrap correctly from both `torch.compile` and DDP wrappers

#### Why
- Matches the Systems & Efficiency lecture guidance so 8× V100 runs saturate hardware without stomping each other's logs or wasting bandwidth on redundant eval/checkpoint ops
- DDP + sampler fixes let each GPU see unique data shards while minimizing NCCL traffic during gradient accumulation, which was the target pattern discussed in EX06/train_ddp.py

#### Remarks
- Launch training with `torchrun --nproc_per_node=8 python main.py --stage train --config ...` to fully leverage the DGX setup

### [2026-03-20] - Gian Seifert

#### What
- Peri-LN (post-sublayer RMSNorm): `x = x + Norm(Module(Norm(x)))` in every TransformerBlock
- Truncated normal initialization: `trunc_normal_(std, a=-3σ, b=3σ)` replacing `normal_`
- WSD LR schedule: warmup → stable plateau → linear decay; replaces cosine annealing
- Z-Loss: `1e-4 * logsumexp(logits)²` auxiliary loss for mixed-precision stability
- New `TrainingConfig` fields: `lr_schedule`, `lr_decay_ratio`, `z_loss_coeff`

#### Why
- Most important why is that we want to be state of the art and this was implemented where possible and feasable. The eaxct changes and decisions can be viewed in the architecture decision md 

#### Remarks
- Existing checkpoints are not loadable after Peri-LN (new norm layer keys in state_dict)

---
### [2026-03-24] - Christof Steiner

#### What
- Integrated Optuna hyperparameter tuning 
- Added `tune.py` CLI entry point with `--resume`, `--export-only`, `--n-trials`, `--timeout` flags
- Added `configs/tune.yaml` with configurable search space (learning rate, weight decay, dropout, d_model, n_layers, n_heads, batch size, etc.)
- Modified `src/training/trainer.py` to accept an optional `trial` parameter for intermediate reporting and pruning
- SQLite-backed study persistence for pause/resume across sessions
- Exports best parameters as `best_params.yaml` for direct use in training configs

#### Why
- Manual hyperparameter selection is suboptimal; Bayesian optimization (TPE) explores the search space more efficiently than grid or random search
- Hyperband pruner kills weak trials early, saving compute 
- Config-driven search space allows reuse across different preprocessing variants without code changes
- SQLite persistence means tuning can survive crashes and be resumed across sessions

#### Remarks
- Run with `python tune.py` (defaults to 50 trials) or `python tune.py --n-trials 100`
- Monitor live with `optuna-dashboard sqlite:///optuna_studies/parrotllm.db`
- Tune once on one preprocessing variant, apply best HPs to all variant comparison runs

### [2026-03-13] - Cyril Gabriele

#### What
- Introduced `configs/project_config.py` plus YAML updates so every stage shares a typed `ProjectConfig`
- Rebuilt `main.py` CLI to load the config once, set the global seed/device, and dispatch stage runners with that single source of truth
- Updated training, evaluation, inference, and chat modules to consume the centralized config/runtime context instead of pulling CLI flags ad hoc
- Hardened seeding via `src/utils.set_seed()` (now also sets `PYTHONHASHSEED` and seeds MPS) and refreshed README docs to explain the new workflow
- Threaded HF token handling through eval/inference so a single HF_TOKEN env var covers all downloads.
- Added stage-specific dummy configs plus a logging schema to speed up smoke tests.
- Tightened preprocessing/tests/utilities to fit the single-source config rules (tokenizer helpers, seed plumbing, etc.).
- Removed every `*.seed` knob from the YAML configs so the single in-code seed remains the only randomness control
- Removed all implicit defaults from `configs/preprocessConfig.py`, `configs/trainingConfig.py`, and the eval/inference/chat sections in `configs/project_config.py`, forcing every YAML entry to spell out the desired values explicitly

#### Why
- Prevent silent divergence between CLI defaults and YAML values by enforcing one configuration source for the full pipeline
- Ensure deterministic experiments by seeding Python/NumPy/PyTorch/MPS exactly once at startup and threading the value through downstream components
- Simplify future maintenance by giving each stage a typed config object rather than loosely structured dictionaries or argparse namespaces
- Explicit configs guarantee no experiment reruns with unintended defaults; Pydantic now rejects runs where a required knob is missing

#### Remarks
- none

### [2026-03-06] - Christof Steiner

#### What
- Added `src/logging_utils.py`: `setup_logger()`, `JSONLLogger`, `make_run_dir()`
- Added `configs/trainingConfig.py`: Pydantic validation for training, model, and logging configs
- Integrated logging into trainer: `train.log` (human-readable) + `metrics.jsonl` (machine-readable)
- Added `tests/test_logging.py` (5 tests)

#### Why
- Logging will help us and the machine to understand and compare the differnet runs.

#### Remarks
- `default.yaml` batch_size=64 causes OOM on MPS; use `test.yaml` for local testing
### [2026-03-05] - Cyril Gabriele

#### What
- Introduced the `--mock-testing` CLI flag and taught inference to load Hugging Face GPT-2 when it's set
- Documented the mock inference workflow in `README.md`
- Let inference pull the sampling temperature from `configs/default.yaml` unless the CLI flag overrides it, fixing the greedy-only bug
- Added a second Hugging Face adapter for the full `openai-community/gpt2` checkpoint and wired `--mock-testing` to use it for more realistic smoke tests
- Added `src/model/hf_gpt2.py` to house the reusable HF GPT-2 adapter
- Auto-load an `HF_TOKEN` from `.env` or the environemnt so mock inference downloads authenticate automatically when available

#### Why
- Enables running inference end-to-end without training or downloading a ParrotLLM checkpoint, which is ideal for quick smoke tests and demos
- Ensures config-driven inference hyperparameters actually take effect without manual flags
- Provides higher-quality mock generations via the standard GPT-2 weights
- Authenticated downloads avoid Hugging Face’s anonymous throttling, making mock inference setup faster by default

#### Remarks
- First run with `--mock-testing` downloads the Hugging Face weights; afterwards they are reused from cache


### [2026-03-04] - Gian Seifert

#### What
- Implemented basic model from lecture with LayerNorm, normal initialization, Multi-Head Attention, learned positional encoding, GELU, dense FFN, dropout and full causal attention
- Added a comprehensive test suite in `tests/model/test_transformer.py` to verify if it works as expected (shape, loss, causality, and parameter count).

#### Why
- Establish a simple baseline model where we can build up from; for this, the lecture recommendations were taken.
- The test case was added to verify if it works as expected and ensure structural integrity.

#### Remarks
- `LayerNorm` parameters were found to include biases even when the global `bias` flag is set to `False` in the config, as `nn.LayerNorm` defaults to `elementwise_affine=True`. The parameter count test was adjusted to account for this.

---

### [2026-03-06] - Tilman Haferbeck

#### What
**Topic classification throughput improvements**
- Input truncation reduced from 512 → 256 characters per document before classification
- Inference batch size increased from 256 → 512
- Both changes applied in `topic_classify_batch()` and the Phase 3.5 main classification loop

**EOS per-document insertion**
- `tokenize_batch()` now appends exactly one EOS token (`50256`) after each document's token list before concatenation into the flat stream
- Matches GPT-2 original pretraining format: `doc_1_tokens + [EOS] + doc_2_tokens + [EOS] + ...`
- Added `truncation=False` explicitly to the `tokenizer()` call to suppress the HuggingFace false-positive warning about sequences longer than 1024 tokens

**Tail-token handling changed from EOS padding to trimming**
- `_pad_to_chunk()` in Phase 8 now trims the tail with `arr[:-remainder]` instead of appending EOS tokens as filler
- EOS is no longer used as structural padding — it is exclusively a semantic document-boundary marker

**Chunked download to avoid OOM on large token budgets**
- Rewrote `download_openwebtext_subset_by_tokens()` in `src/scripts/download_data.py` to flush every 50,000 documents to a temporary Arrow shard on disk instead of accumulating all documents in a Python list
- Shards are concatenated via `concatenate_datasets()` after streaming completes; the temp directory is removed automatically

#### Why
- Halving truncation length and doubling batch size reduces Phase 3.5 runtime by ~2.5× on M4 with no meaningful quality loss — topic is almost always clear from the first 256 characters
- Appending EOS after each document teaches the model a genuine document-boundary signal; without it documents are concatenated raw and the model never learns when one document ends and another begins, causing premature generation termination at inference time
- Using EOS as tail padding corrupts its semantic meaning — the model could learn to predict EOS in positions that are not genuine document boundaries; trimming is correct and the token loss is negligible (< 0.0002% at 700M scale)
- `truncation=False` suppresses a HuggingFace false-positive warning — the tokenizer handles any length correctly; the warning only applies when passing output directly to `model.forward()`, which the pipeline never does
- Accumulating ~1.3M documents as a Python list (~7 GB) causes a silent OOM crash (exit code 137) on machines with ≤ 16 GB RAM; chunked flushing keeps peak RAM at ~250 MB per shard regardless of target token count

#### Remarks
- `truncation=False` must remain set if documents longer than 1024 tokens are to be kept whole — removing it would silently truncate long documents before EOS insertion, breaking the flat-stream chunking assumption
- The chunked download creates a `.tmp_chunks_{seed}/` directory in `data/`; if the process is killed mid-download this directory must be deleted manually before re-running
- Phase 3.5 requires a valid `HF_TOKEN` for the first run to download `textattack/distilbert-base-uncased-ag-news` (~250 MB); subsequent runs load from `~/.cache/huggingface/hub/`

---


### [2026-03-03] - Tilman Haferbeck

#### What
**Seeded subset download**
- Added `download_openwebtext_subset_by_tokens(target_tokens, seed, attrition_rate, avg_tokens_per_doc)` to `src/scripts/download_data.py`
- Estimates required document count as `ceil(target_tokens / avg_tokens_per_doc / (1 - attrition_rate))` and streams exactly that many documents from `Skylion007/openwebtext` using `.shuffle(seed=seed, buffer_size=100_000).take(n_docs)`
- Saves the result to `data/openwebtext-subset-{target_tokens}-seed{seed}/` as a HuggingFace Arrow dataset
- Added `tqdm` progress bar to the streaming loop so the download is visibly alive
- Added `--target-tokens` and `--subset-seed` CLI flags to `src/scripts/download_data.py`

**Topic classification & distribution resampling (Phase 3.5)**
- Added `TOPIC_MODEL_NAME = "textattack/roberta-base-ag-news"` and `TOPIC_LABEL_MAP` constant mapping `LABEL_0–3` to `World`, `Sports`, `Business`, `Sci/Tech`
- Added `_get_topic_pipeline()` with automatic device selection (`mps` > `cuda` > `cpu`) and module-level pipeline cache
- Added `topic_classify_batch()` truncating input to 512 characters before classification
- Phase 3.5 filters documents to the requested topic classes, then optionally resamples within those classes to match a user-specified distribution using `subset_seed` for reproducibility
- Per-class doc count and percentage printed after resampling

**Unified `--topics` CLI argument**
- Added `--topics CLASS[:WEIGHT] [CLASS[:WEIGHT] ...]` to `main.py` replacing the two separate `--topic-classes` and `--topic-distribution` arguments
- Added `parse_topics()` static method to `PreprocessConfig` that splits each token on `:` and returns `(topic_classes, topic_distribution)` — if no weights are provided, `topic_distribution` is `None` and no resampling is performed
- Added `--skip-topic-filter` flag to `main.py` and `PreprocessConfig`
- Added `target_tokens`, `subset_seed`, `topic_classes`, `topic_distribution`, `skip_topic_filter` fields to `PreprocessConfig`

**Tail-token padding (Phase 8)**
- Added `context_length: int = Field(default=1024, ge=1)` to `PreprocessConfig`
- Phase 8 now pads both `train_tokens` and `val_tokens` with EOS token (`50256`) to the nearest exact multiple of `context_length + 1` before writing to disk
- Prints padded token count and complete chunk count for both splits

**Pipeline ordering**
- Moved topic filter from Phase 6.2 to Phase 3.5 (immediately after language detection, before code filter, quality filter, and dedup)

**Progress bars & timing**
- Added per-phase elapsed time (`[Xs]`) to the summary print of every phase
- Added `tqdm` band-level and per-band inner progress bars to `deduplicate_corpus()`
- Added `tqdm` batch-level progress bar to the Phase 3.5 classification loop

**README**
- Added test command (`--target-tokens 10000000`) to Quick Start section (step 3b)
- Added production command (`--target-tokens 1000000000`) with full topic distribution to CLI Reference section

#### Why
- A seeded subset download allows fully reproducible dataset construction without downloading all 38 GB of OpenWebText; only the ~13 shards needed for the target doc count are fetched
- Topic filtering with RoBERTa reduces noise in the training data by keeping only topically relevant documents; placing it at Phase 3.5 means the expensive deduplication and quality phases process a smaller, already-filtered set
- The unified `--topics` argument reduces CLI surface and makes the relationship between class selection and distribution explicit in a single parameter
- Padding the binary output to exact chunk multiples prevents the training data loader's integer division from silently discarding up to `context_length - 1` tokens at the tail of each split


#### Remarks
- `context_length` is hardcoded to `1024` in `PreprocessConfig` as a course requirement; no CLI flag is exposed for it


### [2026-03-01] - Tilman Haferbeck

#### What
**HF caching & memory**
- Set `HF_DATASETS_CACHE=/tmp/parrotllm_hf_cache` via `os.environ.setdefault` before `import datasets` so worker processes (macOS `spawn`) also pick up the redirect
- Added `disable_caching()` call at the start of `run_preprocess()` as a belt-and-suspenders guard
- Added `ds.remove_columns()` after every `.filter()` phase (phases 1–6.1) to drop status/score columns immediately
- Added `shutil.rmtree` cleanup of the temporary HF cache dir at the end of `run_preprocess()`

**Deduplication (Phase 6)**
- Reduced `DEDUP_NUM_PERM` from 64 → 16 and `DEDUP_BANDS` from 16 → 4
- Replaced `blake2b` in `_shingle_hashes()` with Python's built-in `hash()` (~10× faster for non-cryptographic use)
- Rewrote `deduplicate_corpus()`: NumPy array for band hashing (`.tobytes()` instead of tuple hashing), O(n) union-by-first strategy replacing the O(n²) candidate-pair loop and `_jaccard_from_signatures()` call

**Batch throughput**
- Bumped `batch_size` from 256 → 2048 on all nine `.map()` calls (phases 1–7)
- Phase 8 binary output now uses `np.fromiter(itertools.chain.from_iterable(...))` instead of a Python-level extend loop

**Helper function micro-optimisations**
- `_symbol_to_text_ratio`: replaced per-character iteration with `str.count()` per symbol
- `_programming_keyword_count`: replaced `_WORD_SPLIT_RE.findall()` with `text.lower().split()` (no regex overhead)
- `_looks_like_code`: added `stripped[:16]` slice before `.upper()` to avoid uppercasing arbitrary-length lines
- `ellipsis_filter_batch`: replaced `ELLIPSIS_RE.findall()` with `ELLIPSIS_RE.subn()` (single regex pass)
- `heuristic_quality_filter_batch`: reuses the already-computed `words` list; replaced `_longest_word_length(text)` with an inline `max(len(w) for w in words)`; added `_max_ngram_repeats_from_words(words, n)` helper to avoid two redundant `text.split()` calls per document

#### Why
- HuggingFace `datasets` caches every `.map()`/`.filter()` result as Arrow files; on full OpenWebText (~8 M docs, 15 phases) this accumulated >220 GB — redirecting to `/tmp` and deleting on completion keeps the project drive clean
- `disable_caching()` in the main process is not inherited by `spawn` workers on macOS; the env var is the only reliable fix
- The O(n²) pair loop caused Phase 6 to run for 7+ hours on 6 M documents; with `DEDUP_ROWS=4` the LSH band-match already implies high Jaccard similarity so per-pair verification adds no quality benefit
- Halving the hash permutations and bands trades a small drop in recall for a proportional speedup in signature computation and band hashing
- Larger batch sizes reduce Python function-call overhead and improve CPU saturation across all map phases

#### Remarks
- `_max_ngram_repeats` (the text-string variant) is kept for any external callers; internal pipeline now uses `_max_ngram_repeats_from_words`
- The `/tmp` cache guard (`startswith("/tmp")`) prevents accidental deletion if the user overrides `HF_DATASETS_CACHE` to a non-temporary path

### [2026-02-27] - Tilman Haferbeck

#### What
- Added **Section 1c: Extended Text Quality Analysis** to `01_data_preprocessing.ipynb`, comprising 7 new subsections after Section 1b:
  - **1c-1** Punctuation density, uppercase ratio, average line length per document
  - **1c-2** Type-Token Ratio (TTR, computed over first 200 words to remove length bias)
  - **1c-3** URL density (`url/word`) and digit/number ratio per document
  - **1c-4** Paragraph count, ALL-CAPS heading-like lines, and quote density
  - **1c-5** Corpus-level unigram log-probability as a model-free coherence proxy
  - **1c-6** Social media signal and legal boilerplate regex scans
  - **1c-7** 9×9 Pearson cross-signal correlation matrix across all computed signals

#### Why
- To gain a deeper understanding of the dataset's noise profile. Each signal targets a distinct category of low-quality content (markup artifacts, repetitive text, link dumps, incoherent text, scraped boilerplate) and directly informs candidate filter thresholds in `src/data/preprocess.py`

#### Remarks
- The correlation matrix (1c-7) identifies redundant signals before any are promoted to production filters


### [2026-02-26] - Tilman Haferbeck

#### What
- Added ellipsis density filter (Phase 6.1) in `preprocess.py`: documents where `ellipsis_count / word_count > 10 %` are dropped
- Added `ELLIPSIS_RE` regex and `ELLIPSIS_RATIO_THRESHOLD = 0.1` constants
- Exposed `skip_ellipsis_filter` in `PreprocessConfig` and `--skip-ellipsis-filter` CLI flag to bypass the phase

#### Why
- Ellipsis filtering was introduced in the lecture as a standard data quality heuristic and was therefore adopted
- 10 % was chosen as the threshold because legitimate prose may use ellipses occasionally (style, quotations), but a ratio above 10 % reliably indicates truncated, fragmented, or scraped content where ellipses substitute for missing text rather than serving a stylistic purpose

#### Remarks
- None

### [2026-02-26] - Tilman Haferbeck

#### What
- Added **Section 1b: Raw Dataset Statistics** to `01_data_preprocessing.ipynb`: character length and word count descriptives (mean, median, std, min, max, percentiles), short/long document fractions, approximate sentence structure, character composition breakdown (digits, punctuation, whitespace), top 20 most frequent non-stopword words, and 3 random document previews
- Added **Section 4b: Token Sequence Length Distribution**: percentile table (p1–p99), a threshold impact table showing dropped docs and retained token coverage for candidate `min_tokens` cutoffs (16–1024), and an ASCII histogram of token length bins
- Fixed the imports cell: removed `extract_ngrams` and `is_contaminated` from the `src.data.preprocess` import as both are defined locally in notebook cells and do not exist in the production module

#### Why
- Raw statistics give a quick, evidence-backed overview of the dataset before any filtering is applied, making it easier to spot noise and set cleaning thresholds
- Token length analysis directly motivates the `min_tokens = 64` cutoff used in the pipeline by showing how little of the total token budget is lost when dropping very short sequences


#### Remarks
- None

### [2026-02-26] - Cyril Gabriele

#### What
- Added a shared `build_tokenizer` helper that always loads `openai-community/gpt2` with `use_fast=True`, right-side padding, and a dedicated `<|pad|>` token so preprocessing, eval, chat, and inference share identical vocabularies
- Bumped the default tokenizer name in `PreprocessConfig`, increased the model `vocab_size` to 50,258, and stored the corresponding pad/bos/eos ids in `configs/default.yaml`
- Updated tokenizer-dependent unit tests to use the shared helper, ensuring CI exercises the same tokenizer setup as production code

#### Why
- Standardizing on a single pad-aware GPT-2 tokenizer prevents subtle vocabulary drift between preprocessing and runtime components while enabling us to mask padded tokens during loss computation when we switch to packed sequence training

#### Remarks
- Follow-up: ensure the training loop sets `labels=-100` for padded positions once packed batches are introduced


### [2026-02-25] - Cyril Gabriele

#### What
- Introduced a `configs` package exposing a typed `PreprocessConfig` and shared `DEFAULT_LANG` constant for the CLI
- main.py now validates preprocess arguments via `PreprocessConfig` and passes the structured config to `run_preprocess`
- Removed the unused `preprocess_document` helper and doubled the default worker count to `2 * os.cpu_count()` for better CPU saturation
- Documented the default `--lang` behavior in `README.md` so users know when they must supply language models manually

#### Why
- Centralizing defaults keeps the CLI, docs, and pipeline in sync and surfaces invalid preprocessing flags early through Pydantic validation
- Passing a config object simplifies extending the pipeline without modifying the CLI surface every time
- Dropping dead code and increasing worker counts reduces confusion while improving throughput on multi-core hosts
- The README note prevents misconfiguration when someone attempts to target a non-English dataset without the required fastText models

#### Remarks
- Pass `--num-workers` explicitly if `os.cpu_count()` misreports available cores on a host

---

### [2026-02-24] - Christof

#### What
- Moved decontamination from Phase 6 to Phase 1 (runs first on raw text)

#### Why
- Decontamination must run on raw text before sanitization can alter content and change hashes 

#### Remarks
---


---

### [2026-02-19] - Christof

#### What
- Added dual-mode content filtering: code/artifact removal (Phase 3) and quality/coherence filter (Phase 4)
- Two modes via `--filter-mode`: `heuristic` (default, regex/rule-based) and `classifier` (fastText + KenLM)
- Added `--skip-code-filter`, `--skip-quality-filter`, `--filter-mode none` CLI flags
- Added optional `kenlm` dependency for classifier mode
- Created training scripts for classifier models (`train_filter_models.py`, `annotate_educational.py`)
- Raised language detection confidence threshold from 0.5 to 0.8
- Renumbered pipeline phases: dedup→5, decontam→6, tokenize→7, binary→8

#### Why
- We set the threshold in language detection higher since to little documents got removed with 0.5 as threshold
- Added the dual mode since we currently do not know if an LLM like roBERTa/kenlm is allowed als classifier (but LLM approach would be probably be better)
- Added Dedublication because duplicates do not give us any new information



#### Remarks


---

## [2026-02-19] - Project Initialization

### [2026-02-19] - Initial Setup

#### What
- Created project structure (src/, docs/, data/, configs/)
- Implemented transformer architecture in `src/model/transformer.py`
- Added data preprocessing pipeline in `src/data/preprocess.py`
- Set up training infrastructure and evaluation utilities

#### Why
- Established modular foundation for ParrotLLM development
- Enables easier collaboration and maintenance

#### Remarks
- Uses PyTorch framework
- Initial datasets: OpenWebText-10k, WikiText-103
