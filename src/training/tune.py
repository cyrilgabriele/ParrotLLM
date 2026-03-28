"""Optuna hyperparameter tuning runner for ParrotLLM.

Called via: python main.py --stage tune --config configs/tune.yaml
"""

from __future__ import annotations

import copy
import logging
import os
from typing import TYPE_CHECKING

import optuna
import yaml

from configs import ProjectConfig, TuneConfig
from src.utils import get_device

if TYPE_CHECKING:
    pass

log = logging.getLogger("parrotllm.tuning")


# ── HP Sampling ───────────────────────────────────────────────────────────────

def sample_hyperparams(trial: optuna.Trial, search_space: dict) -> dict:
    """Sample hyperparameters from the search space for a single trial."""
    hp = {}
    for name, spec in search_space.items():
        # Support both dict and Pydantic model
        s = spec if isinstance(spec, dict) else spec.model_dump()
        t = s["type"]
        if t == "log_uniform":
            hp[name] = trial.suggest_float(name, s["low"], s["high"], log=True)
        elif t == "uniform":
            hp[name] = trial.suggest_float(name, s["low"], s["high"])
        elif t == "int":
            kwargs = {}
            if s.get("step"):
                kwargs["step"] = s["step"]
            hp[name] = trial.suggest_int(name, s["low"], s["high"], **kwargs)
        elif t == "categorical":
            hp[name] = trial.suggest_categorical(name, s["choices"])
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
        "lr_schedule", "lr_decay_ratio", "z_loss_coeff",
    }

    for k, v in hp.items():
        if k in model_keys:
            cfg["model"][k] = v
        elif k in training_keys:
            cfg["training"][k] = v

    return ProjectConfig.model_validate(cfg)


# ── Objective ─────────────────────────────────────────────────────────────────

def _make_objective(tune_cfg: TuneConfig, base_config: dict):
    """Create the Optuna objective function."""
    # Convert search space to dicts for sample_hyperparams
    search_space = {
        name: spec.model_dump() for name, spec in tune_cfg.search_space.items()
    }

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparams(trial, search_space)
        project_config = build_trial_config(base_config, hp)

        device = get_device(project_config.training.device)
        model_config_dict = project_config.model_dump(mode="python")

        mc = project_config.model
        log.info(f"Trial {trial.number}: lr={hp.get('learning_rate', '?'):.2e}, "
                 f"d_model={mc.d_model}, n_layers={mc.n_layers}, "
                 f"bs={hp.get('batch_size', '?')}, dropout={hp.get('dropout', '?')}")

        from src.training.trainer import run_train
        import torch

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
        except (RuntimeError, torch.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                log.warning(f"Trial {trial.number} OOM — skipping. "
                            f"(bs={hp.get('batch_size')}, accum={hp.get('gradient_accumulation_steps')})")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

        log.info(f"Trial {trial.number} finished: ppl={best_ppl:.2f}")
        return best_ppl

    return objective


# ── Study Creation ────────────────────────────────────────────────────────────

def create_study(tune_cfg: TuneConfig) -> optuna.Study:
    """Create or load an Optuna study with configured sampler and pruner."""
    storage_path = tune_cfg.storage
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    # Sampler
    if tune_cfg.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    elif tune_cfg.sampler == "random":
        sampler = optuna.samplers.RandomSampler()
    else:
        raise ValueError(f"Unknown sampler: {tune_cfg.sampler}")

    # Pruner
    if tune_cfg.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(**tune_cfg.pruner_kwargs)
    elif tune_cfg.pruner == "median":
        pruner = optuna.pruners.MedianPruner(**tune_cfg.pruner_kwargs)
    else:
        raise ValueError(f"Unknown pruner: {tune_cfg.pruner}")

    study = optuna.create_study(
        study_name=tune_cfg.name,
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
        "lr_schedule", "lr_decay_ratio", "z_loss_coeff",
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
            bs = t.params.get("batch_size", "?")
            dp = t.params.get("dropout", "?")
            wd = t.params.get("weight_decay", "?")
            print(f"  #{t.number:>3d}: ppl={t.value:>8.2f} | lr={lr:.2e} | bs={bs} | wd={wd} | dropout={dp}")

    print("=" * 70)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tune(
    project_config: ProjectConfig,
    *,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
    export_only: bool = False,
) -> None:
    """Run Optuna HP tuning using the project configuration."""
    tune_cfg = project_config.tune
    study = create_study(tune_cfg)

    if export_only:
        if len(study.trials) == 0:
            print("No trials found. Run tuning first.")
            return
        print_summary(study)
        export_best_params(study)
        return

    n_trials = n_trials_override or tune_cfg.n_trials
    timeout = timeout_override or tune_cfg.timeout

    # Build base config dict from the model/training/logging sections
    base_config = {}
    if project_config.model:
        base_config["model"] = project_config.model.model_dump(mode="python")
    if project_config.training:
        base_config["training"] = project_config.training.model_dump(mode="python")
    if project_config.logging:
        base_config["logging"] = project_config.logging.model_dump(mode="python")

    log.info(f"Starting Optuna study '{tune_cfg.name}' with {n_trials} trials")

    objective = _make_objective(tune_cfg, base_config)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print_summary(study)
    export_best_params(study)
