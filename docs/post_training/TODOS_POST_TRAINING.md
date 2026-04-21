# Post-Training TODOs

Scope: from today through the pseudo-conference demo (19.05) and final
submission (02.06). Split per the fact sheet: 2 students on SFT, 2 students
on DPO, then reunite for chat interface, poster, and tech report.

Timeline anchors from the fact sheet:
- 13.04 Slide submission
- 14.04 Midterm presentation
- 11.05 Poster submission
- 19.05 Pseudo-Conference (Demo + Poster)
- 02.06 Code + Tech Report submission

---

## Shared (all four) — decide in week 1, before the pairs fork

- [ ] Pick the final pretraining checkpoint to fine-tune from (lowest val
      PPL from the 8B run). Document PPL on Wikitext-103 and OpenWebText
      test in the tech-report baseline table.
- [ ] Agree on the chat template format. Cyril's SFT guide proposes
      `<|system|> / <|user|> / <|assistant|>`. Lock the exact string
      boundaries (newline-separated, no trailing space). Store in
      `configs/default.yaml:tokenizer.chat_template` as the single source
      of truth.
- [ ] Extend tokenizer: add chat special tokens, re-embed their rows
      (zero-init or mean of existing embeddings — pick one, write it down).
      Vocab goes 50 258 → 50 258 + N. Pair A ships this first; Pair B
      must load the exact same tokenizer at DPO time.
- [ ] Define the checkpoint schema change. Add `tokenizer_special_tokens`
      and `training_stage ∈ {base, sft, dpo}` fields. One upgrade script
      for old checkpoints.
- [ ] Directory layout: `runs/sft/…`, `runs/dpo/…`, `runs/eval/…` — both
      pairs agree, nobody stomps on anyone.
- [ ] Midterm presentation (14.04): default is Cyril presents the
      SFT plan, Christof presents the DPO plan, Gian + Tilman present
      pretraining results and the architecture.

---

## Pair A — SFT (Cyril + Tilman)

### Week 1: scaffolding + data

- [ ] Create `src/post_training/sft/` with `data.py`, `collator.py`,
      `trainer.py`. Reuse `src/training/trainer.py` patterns so
      checkpointing, wandb, and device logic are shared.
- [ ] **Tilman**: pick the SFT dataset. Candidates from Cyril's guide:
      `trl-lib/Capybara`, `HuggingFaceH4/ultrachat_200k`. Check license,
      size, language (English only to match pretraining).
- [ ] **Tilman**: run decontamination against the four public leaderboard
      splits (LAMBADA / HellaSwag / Winogrande / OpenBookQA). Reuse
      `src/data/preprocess.py` phase-1 machinery. Report the overlap
      count; if it is above ~1 %, drop the matches.
- [ ] **Cyril**: implement `load_sft_dataset()` — normalize any source
      schema to `{messages: [...]}` per the SFT guide.
- [ ] **Cyril**: implement masking — loss on **assistant tokens only**.
      Mask user/system turns with `-100` in labels. Add a test that
      prints decoded unmasked positions for one example — verify by eye.

### Week 2: SFT production run

- [ ] Hyperparameters (different from pretraining):
      LR ≈ 2e-5 to 5e-5 (~10× lower than pretraining peak),
      warmup ~100 steps, 1–3 epochs, cosine decay, no weight decay on
      the new embedding rows.
- [ ] Tiny-scale smoke test first: 1 000 examples, 200 steps. Confirm
      loss drops and inference on a fixed prompt changes qualitatively.
- [ ] Full SFT run, 1–2 epochs. Save checkpoints tagged
      `training_stage=sft`.
- [ ] Generate sample outputs (10–20 fixed prompts) to
      `docs/post_training/sft_samples.md` for qualitative review and the
      tech report.
- [ ] **Hand the SFT checkpoint to Pair B.** This is the critical
      cross-pair handoff. Expect one round of "this breaks the
      `--leaderboard` contract" feedback.

### Week 3: submit to the official leaderboard + iterate

- [ ] **Tilman**: take the SFT checkpoint and drop it into the
      `submissions/ParrotLLM/runs/` folder of the leaderboard fork.
      Run `uv run python -m leaderboard.run_benchmarks --submission
      ParrotLLM --checkpoint runs/<ckpt>.pt --limit 100`.
- [ ] **Tilman**: commit + push to our fork, open a PR against
      `unisg-ics-dsnlp/PikoGPT_Leaderboard`. GH returns the public
      benchmark scores + the hidden-benchmark total. Record the
      Piko-Intelligence-Index.
- [ ] **Target**: beat the baseline (HellaSwag 33.33, WinoGrande 66.67,
      LAMBADA 0, OpenBookQA 0, Hidden 200, Index 300) with the SFT
      checkpoint. If we do not beat it, that is signal for Pair A to
      iterate on SFT data or hyperparameters before we even get to DPO.
- [ ] Ablate one SFT hyperparameter (LR, dataset size, or epochs) and
      report the delta. Needed for the tech report.

### Week 4: iteration + support for Pair B

- [ ] If time permits, a second SFT run with the better ablation result.
- [ ] Support Pair B's DPO debugging — they are starting from your SFT
      checkpoint, so any oddity (EOS behavior, chat-token embedding
      drift) is yours to diagnose.

### Weeks 5–6: merge, tech report, poster

- [ ] Write the SFT section of the tech report: dataset choice, masking,
      hyperparameters, ablation, samples, benchmark delta.
- [ ] Contribute to poster's "method" half.

---

## Pair B — DPO (Gian + Christof)

### Week 1: prototype loss on the base checkpoint + wire up the official leaderboard repo

- [ ] **Gian**: create `src/post_training/dpo/` with `data.py`,
      `loss.py`, `trainer.py`. Implement the DPO loss from scratch
      (fact sheet forbids frameworks for the main training):

          loss = -log σ( β · (log π_θ(chosen|x)   − log π_ref(chosen|x)
                           − log π_θ(rejected|x) + log π_ref(rejected|x)) )

- [ ] **Gian**: sanity tests for the loss —
      1. If `chosen == rejected`, loss ≈ `-log 0.5` ≈ 0.693, grads ≈ 0.
      2. If the policy is the reference model, loss = 0.693 exactly.
      3. One gradient step must decrease the loss on a fixed batch.
- [ ] **Christof**: fork the course's official leaderboard repo
      <https://github.com/unisg-ics-dsnlp/PikoGPT_Leaderboard> (per
      VL08 slide 26). Create the submission folder:
      `submissions/ParrotLLM/` containing `main.py`, `src/`, and
      `runs/<checkpoint>.pt`. We do **not** build our own harness —
      the course runs this one on the hidden benchmarks.
- [ ] **Christof**: verify our existing `main.py --stage inference` works
      inside the leaderboard repo's expected folder structure. Run the
      smoke test they document:
          `uv run python -m leaderboard.run_benchmarks \
               --submission ParrotLLM --checkpoint runs/<ckpt>.pt \
               --limit 5`
      Output should be a valid results folder with
      `hellaswag/`, `lambada/`, `openbookqa/`, `winogrande/` subfolders.
- [ ] **Christof**: audit the `--leaderboard` flag in our `main.py`.
      VL08 + fact sheet require greedy, seeded, stdout-only output.
      Run twice with the same seed, assert byte-identical output.
      Add `tests/test_leaderboard_contract.py`.

### Week 2: DPO scaffolding + preference data

- [ ] **Gian**: implement the frozen reference-model load path — the
      reference must be loaded in eval mode with `requires_grad=False`
      and its KV activations must not leak into the policy's grads.
      Check with `sum(p.grad for p in ref_model.parameters())` = None
      after `.backward()`.
- [ ] **Christof**: load a preference dataset — candidates:
      `HuggingFaceH4/ultrafeedback_binarized`, `Anthropic/hh-rlhf`.
      Normalize to `{prompt, chosen, rejected}`.
- [ ] **Christof**: extend the checkpoint loader to handle
      `training_stage=dpo` and require that the reference-model
      checkpoint matches `training_stage=sft`.

### Week 3: DPO production run (once Pair A's SFT checkpoint arrives)

- [ ] Receive SFT checkpoint from Pair A. Use it as both initialization
      and reference model (standard DPO setup).
- [ ] Hyperparameters: β ≈ 0.1, LR ≈ 5e-7 to 1e-6, 1 epoch, cosine decay.
- [ ] Monitor policy/reference KL divergence every step. If it
      explodes, lower β or LR. Log it to wandb as a first-class metric.
- [ ] Full DPO run. Save checkpoint tagged `training_stage=dpo`.
- [ ] Generate side-by-side samples (base / SFT / DPO) for the same
      20 prompts used by Pair A — this is the centerpiece of the
      poster.

### Week 4: leaderboard pass + reproducibility lockdown

- [ ] **Christof**: put the DPO checkpoint in
      `submissions/ParrotLLM/runs/` of the leaderboard fork, run the
      full benchmark with `--limit 100`, commit + PR. Record the new
      Piko-Intelligence-Index vs. the SFT baseline.
- [ ] **Christof**: lock down reproducibility for the *project*
      submission (separate from the leaderboard PR) — one-line
      `scripts/submit_benchmarks.sh`, deterministic seeds everywhere,
      `uv.lock` committed. Fact sheet requires a 1-line reproduction.
- [ ] If DPO regressed the benchmarks too hard, run one re-iteration
      with a lower β (standard DPO recovery move) and submit another PR.
      Note: hidden benchmarks can change over time (VL08 slide 29),
      so do not overfit to a single leaderboard snapshot.

### Weeks 5–6: merge, tech report, poster

- [ ] **Gian**: write the DPO section of the tech report — loss
      derivation, β/LR search, KL curves, samples, benchmark delta.
- [ ] **Christof**: write the systems / reproducibility section — CLI
      contract, checkpoint schema, eval harness design, compute used,
      single-command reproduction instructions.
- [ ] Contribute to poster's "method" and "reproducibility" halves.

---

## Reunited (all four) — weeks 5–6

- [ ] Harden `src/chat/app.py` for the demo session. Fact-sheet
      requirements: GUI, text input, multi-turn context, new-chat
      button, checkpoint selector. Gradio is explicitly allowed here
      (only here). Cyril leads, since he already started it.
- [ ] Prepare 10 highlight prompts that show base vs SFT vs DPO
      differences well for the demo.
- [ ] Tech-report merge. Structure:
      1. System (pretraining summary — condensed; point to earlier docs)
      2. SFT (Pair A)
      3. DPO (Pair B)
      4. Evaluation (leaderboard harness + results)
      5. Reproducibility (CLI, seeds, compute accounting)
      6. Limitations and future work
- [ ] Poster merge. Halves: method (Pair A + B) and results
      (benchmarks + samples). One week before the poster submission,
      do one dry-run presentation internally.
- [ ] Final submission (02.06): tag a git release, re-run the
      `--leaderboard` contract test on the final checkpoint, submit
      code + report.

---

## Risks & the interface-drift problem

Things that will bite us if we forget:

- **Special-token tokenizer drift.** If Pair A changes the vocab and
  Pair B loads the old tokenizer, DPO's reference and policy disagree
  token-for-token and every loss value is garbage. The checkpoint must
  carry enough tokenizer state to self-load. Write the loader to
  **assert** vocab agreement at load time; do not warn, fail.
- **SFT/DPO data contamination.** The SFT and preference datasets can
  silently include LAMBADA / HellaSwag / Winogrande / OpenBookQA
  paraphrases. Both pairs dedup against the benchmark test splits
  before every training run.
- **Stdout pollution.** Any library (torch, transformers, datasets) can
  print a warning during model load. In `--leaderboard` mode we need to
  suppress all of them before the generated text is emitted. Re-check
  this after every dependency bump.
- **Coordination cadence.** One weekly 30-minute sync, all four, fixed
  time. Agenda: checkpoint handoffs, interface-contract drift, blockers.
  Single shared wandb project, tagged runs: `base/*`, `sft/*`, `dpo/*`,
  `eval/*`.
