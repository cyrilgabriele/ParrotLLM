"""Optuna hyperparameter tuning for ParrotLLM.

Usage:
    python tune.py                          # uses configs/tune.yaml
    python tune.py --config configs/tune.yaml --n-trials 100
    python tune.py --resume                 # resume previous study
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from pathlib import Path

import optuna
import yaml

from configs import ProjectConfig
from src.logging_utils import init_logging
from src.utils import get_device, set_seed


log = logging.getLogger("parrotllm.tuning")


# ── Config Loading ────────────────────────────────────────────────────────────

def load_tune_config(path: str) -> dict:
    """Load the tuning configuration YAML."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("study", "base_config", "search_space"):
        if key not in cfg:
            raise ValueError(f"tune config missing required section: '{key}'")
    return cfg


# ── HP Sampling ───────────────────────────────────────────────────────────────

def sample_hyperparams(trial: optuna.Trial, search_space: dict) -> dict:
    """Sample hyperparameters from the search space for a single trial."""
    hp = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "log_uniform":
            hp[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "uniform":
            hp[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "int":
            kwargs = {}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"], **kwargs)
        elif t == "categorical":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {t}")

    # ── Constraints ───────────────────────────────────────────────────────────
    # Prune invalid configurations early (lecture pattern: discard, don't patch)
    if "d_model" in hp and "n_heads" in hp:
        if hp["d_model"] % hp["n_heads"] != 0:
            raise optuna.TrialPruned()

    # Derive d_ff from d_model (SwiGLU convention: ~2.67x, rounded to nearest 2)
    if "d_model" in hp:
        raw = int(hp["d_model"] * 8 / 3)
        hp["d_ff"] = raw + (raw % 2)  # round up to even

    # Derive min_lr as 10% of learning_rate
    if "learning_rate" in hp:
        hp["min_lr"] = hp["learning_rate"] * 0.1

    return hp


# ── Config Building ───────────────────────────────────────────────────────────

def build_trial_config(base_config: dict, hp: dict) -> ProjectConfig:
    """Merge sampled hyperparameters into the base config, return ProjectConfig."""
    cfg = copy.deepcopy(base_config)

    model_keys = {"d_model", "n_layers", "n_heads", "d_ff", "dropout"}
    training_keys = {
        "learning_rate", "min_lr", "weight_decay", "beta1", "beta2",
        "grad_clip", "warmup_steps", "batch_size", "gradient_accumulation_steps",
    }

    for k, v in hp.items():
        if k in model_keys:
            cfg["model"][k] = v
        elif k in training_keys:
            cfg["training"][k] = v

    return ProjectConfig.model_validate(cfg)


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, tune_cfg: dict) -> float:
    """Optuna objective: train proxy model, return val perplexity."""
    hp = sample_hyperparams(trial, tune_cfg["search_space"])
    project_config = build_trial_config(tune_cfg["base_config"], hp)

    device = get_device(project_config.training.device)
    model_config_dict = project_config.model_dump(mode="python")

    log.info(f"Trial {trial.number}: lr={hp.get('learning_rate', '?'):.2e}, "
             f"d_model={hp.get('d_model', '?')}, n_layers={hp.get('n_layers', '?')}")

    from src.training.trainer import run_train

    try:
        best_ppl = run_train(
            project_config,
            model_config_dict,
            device=device,
            trial=trial,
        )
    except optuna.TrialPruned:
        log.info(f"Trial {trial.number} pruned.")
        raise

    log.info(f"Trial {trial.number} finished: ppl={best_ppl:.2f}")
    return best_ppl


# ── Study Creation ────────────────────────────────────────────────────────────

def create_study(study_cfg: dict) -> optuna.Study:
    """Create or load an Optuna study with configured sampler and pruner."""
    storage_path = study_cfg.get("storage", "optuna_studies/parrotllm.db")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    # Sampler
    sampler_name = study_cfg.get("sampler", "tpe")
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler()
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Pruner
    pruner_name = study_cfg.get("pruner", "hyperband")
    pruner_kwargs = study_cfg.get("pruner_kwargs", {})
    if pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(**pruner_kwargs)
    elif pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")

    study = optuna.create_study(
        study_name=study_cfg.get("name", "parrotllm-hp-search"),
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )
    return study


# ── Results Export ────────────────────────────────────────────────────────────

def export_best_params(study: optuna.Study, output_path: str = "best_params.yaml") -> None:
    """Export the best trial's parameters as a YAML config snippet."""
    best = study.best_trial
    hp = dict(best.params)

    # Derive d_ff and min_lr
    if "d_model" in hp:
        raw = int(hp["d_model"] * 8 / 3)
        hp["d_ff"] = raw + (raw % 2)
    if "learning_rate" in hp:
        hp["min_lr"] = hp["learning_rate"] * 0.1

    model_keys = {"d_model", "n_layers", "n_heads", "d_ff", "dropout"}
    training_keys = {
        "learning_rate", "min_lr", "weight_decay", "beta1", "beta2",
        "grad_clip", "warmup_steps", "batch_size", "gradient_accumulation_steps",
    }

    result = {"model": {}, "training": {}}
    for k, v in hp.items():
        if k in model_keys:
            result["model"][k] = v
        elif k in training_keys:
            result["training"][k] = v

    with open(output_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    log.info(f"Best params exported to {output_path}")


def print_summary(study: optuna.Study) -> None:
    """Print a summary of the top trials."""
    print("\n" + "=" * 70)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 70)
    print(f"Study: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"Completed: {len(completed)} | Pruned: {len(pruned)} | Failed: {len(failed)}")

    if completed:
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"Best val perplexity: {study.best_value:.2f}")
        print("\nBest hyperparameters:")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")

        print(f"\nTop 5 trials:")
        top5 = sorted(completed, key=lambda t: t.value)[:5]
        for t in top5:
            lr = t.params.get("learning_rate", 0)
            dm = t.params.get("d_model", "?")
            nl = t.params.get("n_layers", "?")
            print(f"  #{t.number:>3d}: ppl={t.value:>8.2f} | lr={lr:.2e} | d_model={dm} | n_layers={nl}")

    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HP tuning for ParrotLLM")
    parser.add_argument("--config", type=str, default="configs/tune.yaml")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of trials from config")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Override timeout (seconds) from config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous study (same as default with SQLite)")
    parser.add_argument("--export-only", action="store_true",
                        help="Just export best params from existing study")
    args = parser.parse_args()

    init_logging(console_level="INFO")
    set_seed(42)

    tune_cfg = load_tune_config(args.config)
    study_cfg = tune_cfg["study"]

    study = create_study(study_cfg)

    if args.export_only:
        if len(study.trials) == 0:
            print("No trials found. Run tuning first.")
            return
        print_summary(study)
        export_best_params(study)
        return

    n_trials = args.n_trials or study_cfg.get("n_trials", 50)
    timeout = args.timeout or study_cfg.get("timeout")

    log.info(f"Starting Optuna study '{study_cfg['name']}' with {n_trials} trials")

    study.optimize(
        lambda trial: objective(trial, tune_cfg),
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print_summary(study)
    export_best_params(study)


if __name__ == "__main__":
    main()
