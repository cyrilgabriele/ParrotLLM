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

### [2026-02-26] - Assistant

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

<!-- Add new entries here -->

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
