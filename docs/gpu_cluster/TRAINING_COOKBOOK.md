# 8xV100 Cluster Training Cookbook

Use this together with `docs/gpu_cluster/TRAINING_TACTIC.md`.

## Assumptions

- `main` is the branch to run. No special commit checkout.
- `configs/default.yaml` on `main` is the final training config.
- Final dataset is `ExperimentC`: `data/exp_c/train.bin` and `data/exp_c/val.bin`.
- Do not preprocess on the cluster.
- Checkpoints are written to `<run_dir>/checkpoints/`, where `training.checkpoint_dir` is resolved relative to the run directory. Always copy the full run directory.

## Step 1: Clone The Repo On The Cluster

Commands:

```bash
git clone git@github.com:cyrilgabriele/ParrotLLM.git
cd ParrotLLM
git branch --show-current
```

Acceptance criteria:

- the repo exists on the cluster
- the current branch is `main`

## Step 2: Copy `ExperimentC` And Point `data/processed` To It

Run from your local machine:

```bash
rsync -avh data/exp_c/ <cluster-user>@<cluster-host>:~/ParrotLLM/data/exp_c/
```

Run on the cluster:

```bash
cd ~/ParrotLLM
mkdir -p data/processed
ln -sfn ../exp_c/train.bin data/processed/train.bin
ln -sfn ../exp_c/val.bin data/processed/val.bin
ls -lh data/exp_c/train.bin data/exp_c/val.bin
ls -lh data/processed/train.bin data/processed/val.bin
```

Acceptance criteria:

- `data/exp_c/train.bin` and `data/exp_c/val.bin` exist and are non-empty
- `data/processed/train.bin` and `data/processed/val.bin` point to `../exp_c/...`

## Step 3: Create The Python Environment

Commands:

```bash
cd ~/ParrotLLM
uv sync --frozen
uv run python -c "import torch; print(torch.__version__)"
```

Acceptance criteria:

- `uv sync --frozen` succeeds
- `import torch` succeeds

## Step 4: Verify GPU Visibility

Commands:

```bash
cd ~/ParrotLLM
nvidia-smi
uv run python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count())"
```

Acceptance criteria:

- `nvidia-smi` shows 8 GPUs
- `cuda_available=True`
- `gpu_count=8`

## Step 5: Run The Single-GPU Smoke Test

Commands:

```bash
cd ~/ParrotLLM
CUDA_VISIBLE_DEVICES=0 uv run python main.py --stage train --config configs/training/train_dummy.yaml
RUN_DIR=$(ls -dt runs/run_* | head -n 1)
find "$RUN_DIR" -maxdepth 1 -type f | sort
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f | sort
```

Acceptance criteria:

- a new `runs/run_*` directory exists
- the run directory contains `config.json`, `train.log`, and `metrics.jsonl`
- at least one checkpoint file exists in `$RUN_DIR/checkpoints/`

## Step 6: Run The 8-GPU DDP Smoke Test

Commands:

```bash
cd ~/ParrotLLM
uv run torchrun --standalone --nproc_per_node=8 main.py --stage train --config configs/gpu_test/ddp_smoke.yaml
RUN_DIR=$(ls -dt runs/run_* | head -n 1)
sed -n '1,80p' "$RUN_DIR/train.log"
```

Acceptance criteria:

- `train.log` contains `distributed=yes`
- `train.log` contains `world_size=8`
- the run finishes without CUDA or NCCL errors

## Step 7: Start The Real Training Run

Commands:

```bash
cd ~/ParrotLLM
uv run torchrun --standalone --nproc_per_node=8 main.py --stage train --config configs/default.yaml
```

Acceptance criteria:

- a new `runs/run_*` directory is created
- `train.log` and `metrics.jsonl` keep growing
- checkpoint files appear in `$RUN_DIR/checkpoints/`

## Step 8: Monitor, Verify, And Copy Out The Run

Commands:

```bash
cd ~/ParrotLLM
RUN_DIR=$(ls -dt runs/run_* | head -n 1)
tail -f "$RUN_DIR/train.log"
```

In a second terminal:

```bash
cd ~/ParrotLLM
RUN_DIR=$(ls -dt runs/run_* | head -n 1)
CKPT=$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'last_*.pt' | sort | tail -n 1)
uv run python -c "import sys, torch; ckpt=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(ckpt['step'])" "$CKPT"
rsync -avh "$RUN_DIR"/ <persistent-path>/$(basename "$RUN_DIR")/
```

Acceptance criteria:

- the log keeps growing during training
- the latest checkpoint loads with `torch.load`
- the full `runs/run_*` directory exists on persistent storage

## Step 9: Resume In The Next 24h Slot

Commands:

```bash
cd ~/ParrotLLM
CKPT=$(find <persistent-run-dir>/checkpoints -maxdepth 1 -type f -name 'last_*.pt' | sort | tail -n 1)
uv run torchrun --standalone --nproc_per_node=8 main.py --stage train --config configs/default.yaml --checkpoint "$CKPT"
```

Acceptance criteria:

- `train.log` contains `resumed from step`
- training continues from the saved checkpoint

## If Something Fails

- If a smoke test fails, first set `training.compile: false` in the config you are running and rerun.
- If worker crashes appear, reduce `training.num_workers` in the config you are running.
- Do not start the full run before Step 6 passes.
