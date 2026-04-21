# Post-Training Team Split (2 + 2)

## What the course mandates

Sources, all locally saved in this directory:
- **VL1** (Exercises Intro): [`NLP_FS26_VL1_roadmap.pdf`](NLP_FS26_VL1_roadmap.pdf)
- **VL08** (Exercise 8, uploaded 21.04): [`NLP_FS26_VL08.pdf`](NLP_FS26_VL08.pdf)
- **Fact sheet**: [`NLP_FS26_factsheet.pdf`](NLP_FS26_factsheet.pdf)

### VL1 — "Roadmap: Building an LLM from Scratch"

The lecture defines four stages and lists the content of each. Under
stage 3, **Post-training**, the three bullets are:

- Instruction tuning
- Alignment trade-offs
- **SFT and DPO**

Stage 4 (Evaluation) is a *separate* stage: "Metric reliability,
Generalization testing, Leaderboard". It is not part of post-training.

### VL1 — "Roadmap: Group Phases"

The group-phase diagram shows the team size per stage:

    Data Preprocessing + EDA → 4 students
    Pretraining               → 4 students
    Post-Training             → 2 + 2 students      ← the split
    Chat-Interface            → 4 students
    Poster                    → 4 students
    Finale Abgabe             → 4 students

The 2+2 split applies to the Post-Training block, and only that block.
Evaluation / Leaderboard runs alongside Post-Training and is the output
of that block, not a separate pair's job.

### Fact sheet (page 1)

> All 4 people work on one codebase with clear separation for the
> post-training separation.

Local copy: [`NLP_FS26_factsheet.pdf`](NLP_FS26_factsheet.pdf).

### VL08 (slide 23) — "Here"

The Exercise-8 deck uploaded today, 2026-04-21, reprints the group-phase
diagram with a green **Here** arrow pointing at the Post-Training block
and a caption: **"Group Phase: 4 → 2×2 | 3 → 3"**. We are at the split
point now.

VL08 slide 24 ("Exercise Today") also tells us what the course is
spending this week on: Poster Template, PikoGPT Leaderboard walkthrough,
and **Theory in Code: RLHF and DPO** (slide 31 confirms: Direct
Preference Optimization).

### The split follows from VL1 + VL08

- VL1 lists Post-training content as **SFT + DPO**.
- VL08 confirms the 2×2 split happens *now*, during Post-Training.
- The exact role of each pair is *not* explicitly spelled out on a
  slide — it is a team decision. But given the course lists exactly
  two post-training methods and requires two pairs, the natural
  mapping is:

      Pair A → SFT   (Instruction tuning)
      Pair B → DPO   (Alignment)

This is a reasoned team decision consistent with the course material,
not something the slides label one-to-one.

## Recommended split

| Pair | Members | Stage | Output |
|------|---------|-------|--------|
| **A — SFT (Instruction Tuning)** | Cyril Gabriele + Tilman Haferbeck | Supervised Fine-Tuning | Instruct checkpoint |
| **B — DPO (Alignment)**         | Gian Seifert + Christof Steiner   | Preference Alignment    | Aligned checkpoint |

Pair B's DPO training starts from Pair A's SFT checkpoint (standard pipeline:
pretrained → SFT → DPO). That handoff is the critical coordination point
between the two pairs.

### Why these pairings (derived from the git history)

**Pair A — SFT (Cyril + Tilman):**

- Cyril has been pushing the codebase toward post-training for two weeks:
  `695d164 added a SFT guide`, `e5f88c5 fixed s.t. demo works`,
  `5a3eaa4 adjusted the chat`. He also owns the tokenizer (`f7464a5`,
  `ada051b`, `9528cd9`), which matters because SFT requires adding chat
  tokens (`<|system|>`, `<|user|>`, `<|assistant|>`) without breaking the
  pretraining vocab of 50 258. He is the natural lead for the SFT design.
- Tilman owns data curation (`4053f0a reworked the preprocessing, right
  tokenization now, bert model for topic classification`,
  `44f7e3d further dataset analysis`, `4001201 statistics on raw and
  tokenized data`) and the preprocessing phase-1 decontamination code.
  SFT needs a curated instruction dataset that is dedup'd against the
  public leaderboard test sets — he already has the tooling.
- Together: Cyril designs the pipeline, Tilman curates the dataset and
  keeps benchmark contamination out of it.

**Pair B — DPO (Gian + Christof):**

- DPO is the harder technical piece in this project. The fact sheet forbids
  high-level frameworks for the main training, so TRL's `DPOTrainer` is
  out — the loss, the frozen reference model, and the β-controlled KL
  have to be implemented from scratch. It is also easy to diverge.
- Gian has the deepest training-loop experience: RMSNorm / SwiGLU / RoPE /
  QK-Norm rewrites (`8183ffc`, `5bc91e7`, `35323ea`, `442c2b7`), the
  distributed Optuna runs (`bc82c10`, `5f7d079`), the cosine→WSD decision
  (`ec3664a`, `6edf5d6`). Numerical stability at small scale is exactly
  what DPO demands.
- Christof brings systems rigor: Muon+AdamW hybrid (`b5af121`), autoresearch
  program (`1a2ba8f`), phased preprocessing (`f486eac`), Azure ML jobs
  (`4b998a4`), logging migration (`902ba28`). DPO needs a clean,
  reproducible runner; he builds that.
- Together: Gian implements and debugs the loss, Christof owns the
  runner, the reference-model loading, and the reproducibility contract
  that Pair B's output has to satisfy for the leaderboard.

### Work shared across both pairs

These belong to both pairs equally and are tracked in the shared section of
the TODO file:

- Leaderboard benchmark harness (LAMBADA / HellaSwag / Winogrande /
  OpenBookQA + hidden). Each pair runs the harness on their own
  checkpoint. The harness itself is written once — since Pair B has the
  stronger CLI-contract discipline (Christof) and eval history (neither
  pair has strong eval-side specialists — Tilman had sliding-window eval
  but that is now inside Pair A), **the harness is split: Christof owns
  the `--leaderboard` CLI contract and test infra; Tilman owns the
  per-benchmark scoring code**. This is the only cross-pair piece of
  code and is fine because it does not block either training pipeline.
- Merging for the chat interface, poster, tech report after post-training
  is done (~week 5).

### Why not other splits

- **SFT and DPO on the same pair, eval on the other pair**: serializes
  badly. The DPO starts from the SFT checkpoint, so if one pair does both
  they block themselves for weeks. Splitting the stages lets Pair B
  prototype the DPO loss on the *base* pretraining checkpoint in parallel
  while Pair A produces the real SFT checkpoint.
- **Cyril + Gian / Tilman + Christof**: puts the two strongest individual
  contributors together, leaves the other pair thin. Splitting them means
  both pairs have one owner of the *training* code and one owner of the
  *data / infra* side. Balanced.

## Inter-pair contracts (decide in week 1)

The codebase stays shared, but these four contracts must be agreed before
the pairs fork so they do not block each other:

1. **Checkpoint schema.** Keep the existing `config + model` payload; add
   `tokenizer_special_tokens` and `training_stage ∈ {base, sft, dpo}`
   fields. Pair A writes the SFT variant first, Pair B mirrors it for DPO.
2. **Tokenizer vocab.** Pair A extends 50 258 → 50 258 + N (chat tokens).
   Pair B must load the *exact same* tokenizer at DPO time — the DPO
   reference model and the policy have to agree token-for-token.
3. **Chat template.** Same string format across SFT, DPO, inference, and
   chat UI. Store the template in `configs/default.yaml:tokenizer.chat_template`
   so there is one source of truth.
4. **Inference CLI.** `python main.py --stage inference --leaderboard` must
   work on *any* of base / sft / dpo checkpoints with no code path change.
   Christof owns this file; anyone else submits PRs.

## TODOs

See [`TODOS_POST_TRAINING.md`](TODOS_POST_TRAINING.md).
