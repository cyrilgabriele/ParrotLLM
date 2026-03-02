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
