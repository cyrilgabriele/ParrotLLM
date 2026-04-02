# LR Scheduler Review

Date: 2026-04-02

Scope:
- Scheduler-related changes across `src/training/trainer.py`, `src/training/tune.py`, config wiring, and focused tests.
- Primary focus was `src/training/trainer.py`.

## Summary

The scheduler core looks sound, and the new scheduler/checkpoint unit tests pass. The main risks are in how the scheduler is integrated into the training loop, especially around skipped optimizer steps, resume behavior, and epoch bookkeeping.

## Findings

### 1. Scheduler can advance when the optimizer did not update

Impact:
- On the fp16 CUDA path, `GradScaler` can skip an optimizer step when gradients overflow.
- The current trainer still calls `scheduler.step()` and increments `completed_steps` even if the model weights were not updated.
- This makes the LR schedule drift ahead of the actual number of weight updates.

Why this matters:
- The run log and checkpoint step count no longer match the true number of optimizer updates.
- Resume continues from an LR position that may be too far ahead.

Notes:
- This does not affect the current `mps` smoke run.
- It matters on the fp16 path that the trainer still supports.

Relevant code:
- `src/training/trainer.py` in the optimizer/scheduler block inside `run_train`

### 2. Resume restores the LR state, but not the dataloader position

Impact:
- Checkpoints restore model, optimizer, scaler, scheduler, and step count.
- They do not restore where training was inside the current epoch or dataloader iterator.
- A resumed run will typically restart data consumption from the beginning of the epoch.

Why this matters:
- Mid-epoch resume is not an exact continuation.
- Some batches are repeated after resume, and the sample order changes relative to the interrupted run.

Notes:
- The LR state is resumed correctly.
- The data stream is not.

Relevant code:
- `src/training/trainer.py` in `load_checkpoint`
- `src/training/trainer.py` where `initial_epoch` is reconstructed and `data_iter = iter(train_loader)` is created

### 3. Epoch boundaries are inferred from integer division, not actual iterator exhaustion

Impact:
- The trainer computes:
  - `steps_per_epoch = len(train_loader) // gradient_accumulation_steps`
- This assumes the loader length is perfectly divisible by `gradient_accumulation_steps`.
- When it is not divisible, the trainer can declare an epoch boundary too early.

Why this matters:
- Evaluation, checkpoint naming, sampler reseeding, and epoch-complete logs can fire before the actual epoch data is fully consumed.
- In DDP, `set_epoch()` can change sampler state before the old iterator is truly exhausted.
- Leftover micro-batches from the old epoch can get mixed with the next epoch's bookkeeping.

Relevant code:
- `src/training/trainer.py` where `steps_per_epoch` is computed
- `src/training/trainer.py` where epoch boundaries are detected from `completed_steps % steps_per_epoch == 0`

### 4. LR config validation happens at runtime, not at config-parse time

Impact:
- Invalid `min_lr`, `lr_schedule`, or `lr_decay_ratio` values are accepted by the Pydantic config model.
- Rejection happens later only when the trainer builds the scheduler.

Why this matters:
- Config errors surface later than necessary.
- Validation would be clearer and safer if it happened during config loading.

Relevant code:
- `configs/training/trainingConfig.py`
- `src/training/trainer.py` in `ParrotLRScheduler.__init__`

## Validation

Passed:
- `uv run pytest tests/training/test_lr_scheduler.py -q`
- `uv run pytest tests/training/test_checkpoint_retention.py -q`

Failed:
- `uv run pytest tests/test_tune.py -q`

Reason for the tune test failures:
- The failures are from stale test assumptions, not from the scheduler math itself.
- The tests still assume the old study name and the old direct architecture parameter shape.
- The current tuning flow now supports `architecture_preset`, which changes what appears in sampled trial params.

## Recommended Follow-up

1. Only advance the scheduler and `completed_steps` when an optimizer update actually happened.
2. Decide whether resume must be exact mid-epoch resume or only step/LR resume, then implement that intentionally.
3. Rework epoch tracking so it follows actual dataloader exhaustion rather than `len(train_loader) // grad_accum`.
4. Add schema-level validation for scheduler-related config fields.
5. Update `tests/test_tune.py` to match the current tuning config and `architecture_preset` behavior.
