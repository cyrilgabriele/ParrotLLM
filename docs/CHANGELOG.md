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
