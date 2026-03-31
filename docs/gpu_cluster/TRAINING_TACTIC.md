# GPU Cluster Training Tactic

This note summarizes the recommended tactic for using the course GPU allocation efficiently.

## Course Constraints

From [NLP_FS26_VL1.pdf](../NLP_FS26_VL1.pdf):

- Pretraining data: provided OpenWebText subset only
- Tokenizer: GPT-2 tokenizer
- Architecture: decoder-only, no MoE
- Context length: `1024`
- Maximum decoder layers: `24`
- Maximum model size: `40M` parameters
- Compute budget: `2 x 24h` on `8x V100` GPUs or similar
- Forbidden: extra pretraining data, external checkpoints, distillation

Operational constraints from the slides:

- V100 does not support Flash Attention
- GPU sessions are deleted after 24h
- The intended use of cluster time is pretraining
- Post-training can run on local hardware or Google Colab

## Recommended Tactic

Treat the two 24-hour slots as one focused pretraining campaign, not as cluster-time exploration.

The guiding idea is:

1. Do architecture and hyperparameter search before the cluster run.
2. Use the cluster almost exclusively for long pretraining runs.
3. Change as little as possible between the two slots.

Given the current repo state, the recommended setup is:

- Model: current `16L / d_model=320 / 8 heads` configuration
- Size: about `35.8M` parameters, safely under the `40M` cap
- Context length: `1024`
- Data recipe: use the current best dataset variant, which is `ExperimentC` according to [dataset_ranking.json](../../results/dataset_eval/dataset_ranking.json)

## Why This Is the Right Bet

This matches the course framing closely:

- You only get `2 x 24h`, so large-scale hyperparameter search on V100s is a poor use of budget.
- The slides explicitly suggest using your own hardware for smaller experiments and coming to the TAs with a concrete concept.
- The slides also explicitly recommend focusing cluster time on pretraining.

This also matches the current repo evidence:

- The model is already within the formal parameter budget.
- The project has DDP support, checkpointing, logging, mixed precision, and `torch.compile`.
- Hyperparameter tuning results already point to a stable region instead of requiring more exploration.
- Dataset evaluation already identifies a leading preprocessing variant.

## Proposed Run Plan

## Launch Command

The training loop lives in `src/training/trainer.py`, but it is launched through `main.py --stage train`.

For the GPU cluster, the command should be:

```bash
uv run torchrun --standalone --nproc_per_node=8 main.py --stage train --config configs/default.yaml
```

To resume from a saved checkpoint in the second 24-hour slot:

```bash
uv run torchrun --standalone --nproc_per_node=8 main.py --stage train --config configs/default.yaml --checkpoint <path-to-checkpoint>
```

Notes:

- `torchrun` is what enables the DDP path inside the trainer
- `--nproc_per_node=8` matches the `8x V100` allocation
- the trainer itself detects distributed mode from the environment and does not need to be called directly
- use a cluster-specific YAML if batch size, checkpoint cadence, or paths differ from `configs/default.yaml`

### Before the Cluster

Finish everything that is cheap to iterate on:

- Freeze the architecture.
- Freeze the dataset variant.
- Freeze the optimizer and LR schedule.
- Make sure preprocessing is complete before the session starts.
- Run a DDP smoke test on a smaller machine before touching the cluster.
- Confirm that checkpoint export from the container is working.

Do not spend V100 time on:

- dataset experimentation
- architecture search
- broad Optuna tuning
- chat UI work
- post-training experiments

### Run 1

Use the first slot as a calibration run that can continue into a full run if everything looks healthy.

Suggested flow:

1. Start with the final model and final dataset.
2. Measure:
   - actual tokens/sec
   - memory headroom
   - effective step time
   - validation-loss behavior in the first part of training
3. Convert the measured throughput into a realistic `max_steps` for the full 24h window.
4. If the loss curve is healthy, continue the same run instead of restarting.

Important principle:

- Do not keep `max_steps` arbitrary.
- Set it from measured throughput so that the warmup and WSD decay happen at the right point inside the actual 24h budget.

### Run 2

Use the second slot to continue the best pretraining checkpoint from Run 1.

Default choice:

- resume and train longer

Only deviate if Run 1 shows a clear, isolated issue such as:

- unstable loss
- under-utilized GPU memory that can be turned into a larger batch
- checkpoint cadence being too sparse

If you change anything for Run 2, change exactly one variable and keep the rest fixed.

## Operational Recommendations

### Checkpointing

The session is deleted after 24h, so checkpointing must be much more aggressive than a casual local run.

Recommendations:

- save roughly hourly, not just every several thousand steps
- verify that checkpoints are copied out of the container during the run
- keep a final checkpoint and at least a few intermediate recovery points

### Throughput First

On V100s, training efficiency matters more than ambitious features.

Priorities:

- rely on standard PyTorch scaled dot-product attention, not Flash Attention
- prefer stable DDP execution over clever but fragile changes
- use `torch.compile` only if it is already proven stable in your environment
- keep logging and evaluation frequent enough to detect failure, but not so frequent that they steal training time

### Reproducibility

- fix seeds
- log exact configs
- log train/validation loss over time
- keep one command that launches training end-to-end

## Main Risks To Address Before Booking Time

### 1. Tokenizer Compliance

The course slide says GPT-2 tokenizer. The current repo appears to add a dedicated pad token and uses vocabulary size `50258`, not plain GPT-2 `50257`.

That may still be acceptable, but it should be clarified with the TAs before final runs if the requirement is meant literally.

### 2. Checkpoint Loss

The cluster session is ephemeral. A run without validated checkpoint export is too risky.

### 3. Misaligned Schedule

If `max_steps`, warmup, and decay are not tied to real cluster throughput, the run may spend too much or too little of the budget in the wrong LR phase.

### 4. Spending Cluster Time on the Wrong Stage

The course guidance strongly suggests:

- cluster for pretraining
- local/Colab for post-training

## Bottom Line

The highest-probability tactic is:

- freeze the current model family
- use the best current dataset recipe
- spend Run 1 on calibration plus real pretraining
- spend Run 2 on continuing the best checkpoint
- avoid using cluster time for search

If needed, this can be turned into a concrete execution checklist with proposed values for:

- `batch_size`
- `gradient_accumulation_steps`
- `max_steps`
- warmup length
- checkpoint cadence
- go/no-go criteria for the first hour of Run 1
