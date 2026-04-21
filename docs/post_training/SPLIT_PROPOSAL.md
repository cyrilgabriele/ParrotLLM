# Post-Training Split — Proposal

## The split

| Pair | Members | Method | Output |
|------|---------|--------|--------|
| **A** | Cyril + Tilman | **SFT** (Instruction Tuning) | Instruct checkpoint |
| **B** | Gian + Christof | **DPO** (Alignment)         | Aligned checkpoint |

Pair B's DPO starts from Pair A's SFT checkpoint.

## One-line reason per person

- **Cyril → SFT.** Already wrote the SFT guide (`docs/post_training/sft_from_checkpoint_summary.md`, 14.04). Owns the tokenizer.
- **Tilman → SFT.** Data curation + decontamination experience from preprocessing work.
- **Gian → DPO.** Deepest training-loop expertise (custom loss from scratch, numerical stability).
- **Christof → DPO.** 5090 at home → fits policy + frozen reference in VRAM. Also owns CLI/infra for the leaderboard submission.

## Course-mandated (not negotiable)

- 2 + 2 split during post-training — fact sheet p. 1, VL1 "Group Phases", VL08 slide 23.
- Both SFT and DPO must ship — VL1 "Roadmap" stage 3.
- Reunite after post-training for chat UI, poster, tech report.

## Team decision (what we're discussing)

1. **Which pair gets which method?** (Proposal above.)
2. **Benchmark harness:** use the course's `unisg-ics-dsnlp/PikoGPT_Leaderboard` repo (fork → PR flow). Confirmed not building our own.
3. **Baseline to beat:** Piko-Intelligence-Index 300 (VL08 slide 29).

## Alternative if someone objects

Swap Tilman ↔ Christof:
- Pair A SFT: Cyril + Christof
- Pair B DPO: Gian + Tilman

Still works, but loses Christof's 5090 on the GPU-heavier DPO side.

## Next step

Agree on the split today, then each pair picks up its week-1 tasks from
`TODOS_POST_TRAINING.md`.
