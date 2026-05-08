# DPO β-down (β=0.05) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a new submission checkpoint by running continuation-preference DPO with β=0.05 (half the current β=0.1) on top of the existing SFT-benchmark base, then benchmark and decide whether to swap into the submission.

**Architecture:** One new YAML config (copy of `dpo_continuation.yaml` with two edits), reusing the existing prepared 24,500 continuation pairs. Train via `main.py --stage dpo`, pick the lowest-loss `best_loss_*.pt`, bench at `--limit 200` against the same four leaderboard tasks. No source-code changes.

**Tech Stack:** Python 3.11+, PyTorch (MPS backend on Apple Silicon), `uv` for command execution, the project's existing DPO trainer, the leaderboard subprocess at `external/PikoGPT_Leaderboard`.

**Key facts** (verified against the codebase):
- Submission checkpoint is at `Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt`, scored 22.0 / 58.5 / 36.5 / 36.5 (sum 153.5) at limit=200.
- The DPO trainer reads `dpo.beta` from YAML; no code change required to lower it.
- Prepared pairs at `data/posttraining/dpo_pairs_continuation/` (manifest exists) — `dpo-prepare` is a no-op when the manifest is present.
- The leaderboard runner writes per-checkpoint overview JSON to `external/PikoGPT_Leaderboard/Results/parrotlabs_parrotllm/<ckpt_stem>/parrotlabs_parrotllm__<ckpt_stem>__overview.json`.
- All Python invocations use `uv run` per project convention.
- Commits do **not** include a `Co-Authored-By` line.

---

## File Structure

- **Create:** `configs/posttraining/dpo_continuation_beta005.yaml` — new DPO config, two-line delta vs. `dpo_continuation.yaml`.

That's it. No source files modified, no new tests. The change is data, not behavior — the DPO trainer code already supports the β knob.

---

### Task 1: Create the β=0.05 DPO config

Single new file. Two-line delta vs. `dpo_continuation.yaml`: the `beta` value and the `runs_dir`. Everything else (reference checkpoint, prepared_dir, sources, lr, epochs) is preserved verbatim.

**Files:**
- Create: `configs/posttraining/dpo_continuation_beta005.yaml`

- [ ] **Step 1.1: Copy the existing config**

Run:
```bash
cp /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/configs/posttraining/dpo_continuation.yaml \
   /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/configs/posttraining/dpo_continuation_beta005.yaml
```

- [ ] **Step 1.2: Edit the two fields**

In `configs/posttraining/dpo_continuation_beta005.yaml`:

Replace the `runs_dir` line:
```yaml
  runs_dir: runs/posttraining/dpo_continuation
```
with:
```yaml
  runs_dir: runs/posttraining/dpo_continuation_beta005
```

Replace the `beta` line:
```yaml
  beta: 0.1
```
with:
```yaml
  beta: 0.05
```

(The leading two spaces are part of the YAML indentation under `dpo:` — preserve exactly.)

- [ ] **Step 1.3: Update the leading comment block**

The first three comment lines under `dpo:` describe the original recipe. Replace:
```yaml
  # Continuation-preference DPO recipe (Plan B).
  # `reference_checkpoint` is set by the calling overnight script after the
  # SFT run finishes. Substitute the literal placeholder string below.
```
with:
```yaml
  # Continuation-preference DPO with halved beta (β=0.05 vs the 0.1 in
  # dpo_continuation.yaml). Same reference, same 24,500 prepared pairs,
  # same lr/epochs — single-axis test of "push DPO harder" per the TA hint
  # that hidden-benchmark gains may justify some LAMBADA loss.
  # See docs/superpowers/specs/2026-05-08-dpo-betadown-design.md.
```

- [ ] **Step 1.4: Verify the YAML loads and the values are correct**

Run:
```bash
cd /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM && uv run python -c "
import yaml
cfg = yaml.safe_load(open('configs/posttraining/dpo_continuation_beta005.yaml'))
dpo = cfg['dpo']
assert dpo['beta'] == 0.05, f'beta is {dpo[\"beta\"]}'
assert dpo['runs_dir'] == 'runs/posttraining/dpo_continuation_beta005', f'runs_dir is {dpo[\"runs_dir\"]}'
assert dpo['preference_format'] == 'continuation'
assert dpo['learning_rate'] == 2.0e-6
assert dpo['num_epochs'] == 1.0
assert dpo['reference_checkpoint'].endswith('best_loss_0p9853_epoch_0000_step_0000300.pt')
print('OK: beta=0.05, runs_dir=runs/posttraining/dpo_continuation_beta005, ref unchanged')
"
```
Expected output: `OK: beta=0.05, runs_dir=runs/posttraining/dpo_continuation_beta005, ref unchanged`

If anything else printed (e.g. AssertionError), re-check the edits in Steps 1.2–1.3.

- [ ] **Step 1.5: Verify the prepared pairs are still in place (no `dpo-prepare` needed)**

Run:
```bash
test -f /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/data/posttraining/dpo_pairs_continuation/manifest.json && \
  echo "OK: pairs already prepared" || echo "MISSING: need to run dpo-prepare"
```
Expected: `OK: pairs already prepared`

If MISSING, that's a regression — investigate before continuing (the lengthnorm run reused the same dir, so the manifest should exist).

---

### Task 2: Run DPO training

Single command. Long-running (~25–40 min on the user's MPS rig at 24,500 pairs / batch 4 ≈ 6,125 steps for 1 epoch). The trainer writes checkpoints to `runs/posttraining/dpo_continuation_beta005/run_<TIMESTAMP>/checkpoints/`.

**Files:**
- (No edits — this task only runs commands.)

- [ ] **Step 2.1: Confirm no other heavy MPS process is running**

Run:
```bash
ps aux | grep -E "main\.py|run_benchmarks|dpo" | grep -v grep
```
Expected: no rows, or only this `grep` itself. If a prior training/bench is still running, stop it before starting (the 42 GiB MPS ceiling is shared and the previous bench just OOM'd).

- [ ] **Step 2.2: Start training in the background**

Run:
```bash
cd /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM && \
  mkdir -p runs/dpo_betadown_logs && \
  uv run python main.py --stage dpo --config configs/posttraining/dpo_continuation_beta005.yaml \
  > runs/dpo_betadown_logs/train.log 2>&1 &
echo "PID: $!"
```
Expected: prints a PID. Save it; you'll watch the log next.

- [ ] **Step 2.3: Wait for training to finish; tail the log periodically**

The training script prints per-step log lines into `runs/dpo_betadown_logs/train.log` and writes the run dir name early. While it runs, monitor by tailing every few minutes:

```bash
tail -3 /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/runs/dpo_betadown_logs/train.log
```

Training is finished when the file's final `step` value approaches the manifest's `n_train` count (~6,125 for batch 4) **and** the process is gone:
```bash
ps -p <PID> > /dev/null && echo "still running" || echo "finished"
```

- [ ] **Step 2.4: Verify checkpoints were written**

Run:
```bash
ls /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/runs/posttraining/dpo_continuation_beta005/run_*/checkpoints/
```
Expected: at least one `best_loss_*.pt` and one `last_*.pt`. If empty, read the tail of `train.log` for the failure cause.

---

### Task 3: Pick the best checkpoint by loss

Mirror of the picker in `scripts/overnight_dpo_compare.sh`. Lowest-loss `best_loss_*.pt` wins.

**Files:**
- (No edits — this task only runs a command.)

- [ ] **Step 3.1: Print the chosen checkpoint path**

Run:
```bash
cd /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM && uv run python -c "
import glob, os
candidates = sorted(glob.glob('runs/posttraining/dpo_continuation_beta005/run_*/checkpoints/best_loss_*.pt'))
def loss_key(p):
    base = os.path.basename(p)
    val = base.split('_')[2].replace('p', '.')
    return float(val)
best = min(candidates, key=loss_key)
print(os.path.abspath(best))
"
```
Expected: an absolute path to the lowest-loss checkpoint. Save this string for Task 4.

---

### Task 4: Benchmark the chosen checkpoint at limit=200

Same invocation pattern as `scripts/overnight_dpo_compare.sh`'s `bench_best`. Runs 4 sub-benchmarks sequentially. Total time ~10–15 min.

**Files:**
- (No edits — this task only runs commands.)

- [ ] **Step 4.1: Run the benchmark**

Replace `<CKPT>` with the path printed in Task 3.

```bash
cd /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/external/PikoGPT_Leaderboard && \
  uv run python -m leaderboard.run_benchmarks \
    --submission parrotlabs_parrotllm \
    --submissions-dir /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/Submissions \
    --checkpoint <CKPT> \
    --python /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/.venv/bin/python3 \
    --limit 200 \
  > /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/runs/dpo_betadown_logs/bench.log 2>&1
```

Expected: process exits 0. If it OOMs (file is short or empty), close other apps and retry — same risk that killed the lengthnorm step-6000 bench.

- [ ] **Step 4.2: Print the four scores**

```bash
grep -E "^(hellaswag|winogrande|openbookqa|lambada):" \
  /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/runs/dpo_betadown_logs/bench.log
```
Expected: four lines, e.g.
```
hellaswag: 45/200 (22.50%)
winogrande: 119/200 (59.50%)
openbookqa: 73/200 (36.50%)
lambada: 64/200 (32.00%)
```

Compute the sum and note it for Task 5.

---

### Task 5: Decide and (if it wins) swap into the submission

The decision rule from the spec:

| Condition | Action |
|---|---|
| sum > 153.5 AND LAMBADA ≥ 32 | Swap into submission |
| sum > 153.5 AND LAMBADA < 28 | Keep current submission; queue a β=0.07 follow-up |
| sum ≤ 153.5 | Discard; queue Path B (length-norm + β=0.02) |

**Files (only if swapping):**
- Modify: `Submissions/parrotlabs_parrotllm/runs/MANIFEST.md`

- [ ] **Step 5.1: Write the comparison row**

Append to `runs/overnight_dpo_compare.md` (or echo to stdout if you don't want to edit that file):
```markdown
| dpo_continuation_beta005 (β=0.05) | <HS> | <WG> | <OBQA> | <LAMBADA> | <SUM> |
```
filling in the numbers from Task 4.2.

- [ ] **Step 5.2: Apply the decision rule**

If `sum ≤ 153.5`: stop here. The experiment failed to beat baseline; pause and report the four scores back to the user before doing anything else. Do **not** modify the submission.

If `sum > 153.5 AND LAMBADA < 28`: stop here. LAMBADA collapsed past the spec's tolerance; report the scores and propose the β=0.07 follow-up. Do **not** modify the submission.

Otherwise (sum > 153.5 AND LAMBADA ≥ 32): proceed to Step 5.3.

- [ ] **Step 5.3: Compute the new checkpoint's SHA-256 and size**

Replace `<CKPT>` with the path from Task 3.

```bash
shasum -a 256 <CKPT>
ls -la <CKPT> | awk '{print $5}'
```

Save the SHA and the byte count for the manifest update.

- [ ] **Step 5.4: Copy the checkpoint into the submission folder**

```bash
cp <CKPT> /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt
```

- [ ] **Step 5.5: Update the submission manifest**

In `Submissions/parrotlabs_parrotllm/runs/MANIFEST.md`, replace the entire `## File metadata` and `## Provenance` and `## Bench results at limit=200` sections with the new values:

```markdown
## File metadata

- **Filename:** `parrotlabs_final.pt`
- **Size:** ~458 MB (<NEW_BYTES> bytes)
- **SHA-256:** `<NEW_SHA>`
- **Format:** PyTorch state_dict + config bundle (`torch.load` compatible)

## Provenance

- **Source run:** `runs/posttraining/dpo_continuation_beta005/run_<TIMESTAMP>/`
- **Source file:** `checkpoints/<BASENAME_OF_CKPT>`
- **Stage:** continuation-pair DPO (Plan B), 1 epoch, **β=0.05**, lr=2.0e-6
- **Reference:** SFT checkpoint at
  `runs/posttraining/sft_benchmark/run_20260506_010816_sft_lr_5e-07/checkpoints/best_loss_0p9853_epoch_0000_step_0000300.pt`
  (Alpaca template, lr=5e-7, early-stopped at step 300)
- **Best DPO step:** <STEP_NUMBER> (training loss <LOSS_FROM_FILENAME> from SFT 0.985 from base 6.79)

## Bench results at limit=200

| Benchmark | Score |
|---|---|
| HellaSwag  | <HS_PCT>% |
| WinoGrande | <WG_PCT>% |
| OpenBookQA | <OBQA_PCT>% |
| LAMBADA    | <LAMBADA_PCT>% |
| **Average** | **<AVG>%** |
```

Substitute every `<...>` placeholder with real values before writing.

- [ ] **Step 5.6: Commit the new config, the manifest update, and the spec/plan**

```bash
cd /Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM && \
  git add configs/posttraining/dpo_continuation_beta005.yaml \
          Submissions/parrotlabs_parrotllm/runs/MANIFEST.md \
          docs/superpowers/specs/2026-05-08-dpo-betadown-design.md \
          docs/superpowers/plans/2026-05-08-dpo-betadown.md && \
  git commit -m "feat(dpo): β=0.05 continuation-DPO is the new submission

New best at limit=200: HS <HS>% / WG <WG>% / OBQA <OBQA>% / LAMBADA <LAMBADA>%
(sum <SUM>, vs prior 153.5).
Same SFT base, same 24,500 continuation pairs, β halved from 0.1.
Spec: docs/superpowers/specs/2026-05-08-dpo-betadown-design.md"
```

(Do **not** add a `Co-Authored-By` line.)

- [ ] **Step 5.7: Re-upload the checkpoint to Hugging Face**

This is a manual user action — the local file is what's tracked, but the public submission also lives at `huggingface.co/ParrotLabs/parrotlabs_parrotllm`. Surface this to the user as a TODO with the new SHA-256 they should verify after upload.

---

## Self-Review

**1. Spec coverage:**
- Spec §"Approach (selected: Path A)" → Task 1 (config), Task 2 (training).
- Spec §"Execution" steps 1–4 → Tasks 1–4 directly.
- Spec §"Execution" step 5 (decision rule) → Task 5.
- Spec §"Risks" (collapse, OOM) → mitigated by Task 2.1 (process check) and Task 4.1 (retry-on-OOM note).
- Spec §"Out of scope" → respected: no SFT changes, no new data prep, no inference changes, no length-norm.

**2. Placeholder scan:** The plan uses concrete commands and values. The two `<CKPT>` and several `<...>` placeholders in Task 5 are *runtime* substitutions (not authoring placeholders) — they're filled with values produced by earlier tasks. No "TBD" / "TODO" / "implement later" anywhere.

**3. Type consistency:** No new types are introduced. The plan only references existing config fields (`beta`, `runs_dir`, `learning_rate`, `num_epochs`, `reference_checkpoint`, `preference_format`) and existing CLI flags (`--stage dpo`, `--config`, `--limit`, `--checkpoint`, `--submission`, `--submissions-dir`, `--python`). All verified against `dpo_continuation.yaml` and `scripts/overnight_dpo_compare.sh`.
