"""Evaluate dataset quality by training with identical HPs on each dataset.

Uses the best hyperparameters from Optuna study (Trial #8) and trains
the proxy model (3000 steps, context 256) on each dataset A-F.
The dataset with the lowest validation perplexity has the best quality.

Usage:
    python eval_datasets.py ExperimentA ExperimentB
    python eval_datasets.py --no-compile ExperimentC
    python eval_datasets.py ExperimentA ExperimentB ExperimentC ExperimentD ExperimentE ExperimentF
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml


DATASETS = {
    "ExperimentA": ("data/exp_a/train.bin", "data/exp_a/val.bin"),
    "ExperimentB": ("data/exp_b/train.bin", "data/exp_b/val.bin"),
    "ExperimentC": ("data/exp_c/train.bin", "data/exp_c/val.bin"),
    "ExperimentD": ("data/exp_d/train.bin", "data/exp_d/val.bin"),
    "ExperimentE": ("data/exp_e/train.bin", "data/exp_e/val.bin"),
    "ExperimentF": ("data/exp_f/train.bin", "data/exp_f/val.bin"),
}

CONFIG_PATH = "configs/tuning/dataset_eval.yaml"
RESULTS_DIR = "results/dataset_eval"


def run_single_dataset(name: str, train_bin: str, val_bin: str, compile: bool) -> dict:
    """Train on one dataset and return metrics."""
    import torch
    from configs import ProjectConfig
    from src.training.trainer import run_train
    from src.utils import get_device

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["train_bin"] = train_bin
    cfg["training"]["val_bin"] = val_bin
    # torch.compile only works on CUDA (Linux) — disable for MPS/CPU
    cfg["training"]["compile"] = compile and torch.cuda.is_available()

    project_config = ProjectConfig.model_validate(cfg)
    device = get_device(project_config.training.device)
    model_config_dict = project_config.model_dump(mode="python")

    print(f"\n{'='*60}")
    print(f"  DATASET: {name}")
    print(f"  train: {train_bin} ({Path(train_bin).stat().st_size / 1e6:.1f} MB)")
    print(f"  val:   {val_bin} ({Path(val_bin).stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}\n")

    t0 = time.time()
    best_ppl = run_train(
        project_config,
        model_config_dict,
        device=device,
    )
    elapsed = time.time() - t0

    # Clear GPU memory between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "dataset": name,
        "val_perplexity": round(best_ppl, 4),
        "train_bin": train_bin,
        "train_size_mb": round(Path(train_bin).stat().st_size / 1e6, 1),
        "duration_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset quality evaluation")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (for Windows)")
    parser.add_argument("datasets", nargs="+", choices=list(DATASETS.keys()),
                        help="Datasets to evaluate (e.g. ExperimentA ExperimentB)")
    args = parser.parse_args()

    compile = not args.no_compile
    datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}

    # Verify all datasets exist
    for name, (train, val) in datasets.items():
        assert Path(train).exists(), f"Missing: {train}"
        assert Path(val).exists(), f"Missing: {val}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "dataset_ranking.json")

    # Load existing results from previous runs
    existing = {}
    if Path(output_path).exists():
        with open(output_path) as f:
            for entry in json.load(f):
                existing[entry["dataset"]] = entry

    for name, (train_bin, val_bin) in datasets.items():
        result = run_single_dataset(name, train_bin, val_bin, compile)
        existing[name] = result
        print(f"\n>>> {name}: val_ppl = {result['val_perplexity']:.2f} "
              f"({result['duration_seconds']:.0f}s)\n")

        # Save after each dataset so no results are lost
        all_results = sorted(existing.values(), key=lambda r: r["val_perplexity"])
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary of all results so far
    all_results = sorted(existing.values(), key=lambda r: r["val_perplexity"])
    print("\n" + "=" * 60)
    print(f"DATASET QUALITY RANKING ({len(all_results)}/6 completed)")
    print("=" * 60)
    for i, r in enumerate(all_results):
        marker = " <<< BEST" if i == 0 else ""
        print(f"  {i+1}. {r['dataset']:>12s}: ppl={r['val_perplexity']:>8.2f} "
              f"({r['train_size_mb']:.0f} MB, {r['duration_seconds']:.0f}s){marker}")
    print("=" * 60)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
