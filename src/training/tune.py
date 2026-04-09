"""Optuna hyperparameter tuning runner for ParrotLLM.

Called via: python main.py --stage tune --config configs/tune.yaml
"""

from __future__ import annotations

import copy
from itertools import product
import logging
import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import optuna
import yaml

from configs import ProjectConfig, TuneConfig
from src.utils import get_device

if TYPE_CHECKING:
    pass

log = logging.getLogger("parrotllm.tuning")
ARCHITECTURE_KEYS = {"d_model", "n_layers", "n_heads", "d_ff"}


# ── HP Sampling ───────────────────────────────────────────────────────────────

def _derive_d_ff(d_model: int) -> int:
    """Derive a SwiGLU FFN width from the model width."""
    raw = int(d_model * 8 / 3)
    return raw + (raw % 2)


def estimate_model_params(
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
) -> int:
    """Estimate unique trainable parameters for the current transformer sketch."""
    d_head = d_model // n_heads
    embed = vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model + 2 * d_head
    return embed + n_layers * per_layer


def _expand_discrete_values(name: str, spec: dict) -> list[int]:
    """Expand an int/categorical search spec into explicit candidate values."""
    spec_type = spec["type"]
    if spec_type == "categorical":
        return [int(v) for v in spec["choices"]]
    if spec_type == "int":
        step = int(spec.get("step") or 1)
        return list(range(int(spec["low"]), int(spec["high"]) + 1, step))
    raise ValueError(
        f"Architecture search requires '{name}' to use 'int' or 'categorical', got '{spec_type}'."
    )


def _encode_architecture_preset(*, d_model: int, n_layers: int, n_heads: int, d_ff: int) -> str:
    """Encode an architecture preset as a single Optuna categorical choice."""
    return f"d_model={d_model},n_layers={n_layers},n_heads={n_heads},d_ff={d_ff}"


def _decode_architecture_preset(preset: str) -> dict[str, int]:
    """Decode an Optuna architecture preset string into model hyperparameters."""
    values: dict[str, int] = {}
    for item in preset.split(","):
        key, raw = item.split("=", maxsplit=1)
        values[key] = int(raw)
    return values


def _resolve_trial_params(trial: optuna.trial.FrozenTrial | optuna.Trial) -> dict:
    """Expand stored Optuna params into the concrete model/training values."""
    hp = dict(trial.params)
    if "architecture_preset" in hp:
        hp.update(_decode_architecture_preset(hp.pop("architecture_preset")))
    if "d_model" in hp and "d_ff" not in hp:
        hp["d_ff"] = _derive_d_ff(hp["d_model"])
    if "learning_rate" in hp and "min_lr" not in hp:
        hp["min_lr"] = hp["learning_rate"] * 0.1
    return hp


def _build_architecture_presets(
    search_space: dict,
    *,
    vocab_size: int,
    param_budget_min: int,
    param_budget_max: int,
) -> list[str]:
    """Generate all valid architecture presets inside the configured parameter window."""
    d_model_values = _expand_discrete_values("d_model", search_space["d_model"])
    n_layer_values = _expand_discrete_values("n_layers", search_space["n_layers"])
    n_head_values = _expand_discrete_values("n_heads", search_space["n_heads"])
    if "d_ff" in search_space:
        d_ff_values = _expand_discrete_values("d_ff", search_space["d_ff"])
    else:
        d_ff_values = None

    budget_midpoint = (param_budget_min + param_budget_max) / 2
    presets: list[tuple[float, str]] = []
    for d_model, n_layers, n_heads in product(d_model_values, n_layer_values, n_head_values):
        if d_model % n_heads != 0:
            continue
        d_head = d_model // n_heads
        if d_head % 2 != 0:
            continue

        candidate_d_ff_values = d_ff_values if d_ff_values is not None else [_derive_d_ff(d_model)]
        for d_ff in candidate_d_ff_values:
            n_params = estimate_model_params(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
            )
            if param_budget_min <= n_params <= param_budget_max:
                preset = _encode_architecture_preset(
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_ff=d_ff,
                )
                presets.append((abs(n_params - budget_midpoint), preset))

    presets.sort(key=lambda item: item[0])
    return [preset for _, preset in presets]


def sample_hyperparams(
    trial: optuna.Trial,
    search_space: dict,
    *,
    model_defaults: dict | None = None,
    param_budget_min: int | None = None,
    param_budget_max: int | None = None,
) -> dict:
    """Sample hyperparameters from the search space for a single trial."""
    hp = {}
    use_architecture_presets = (
        param_budget_min is not None
        and param_budget_max is not None
        and {"d_model", "n_layers", "n_heads"}.issubset(search_space)
    )
    if use_architecture_presets:
        vocab_size = int((model_defaults or {}).get("vocab_size", 50258))
        presets = _build_architecture_presets(
            search_space,
            vocab_size=vocab_size,
            param_budget_min=param_budget_min,
            param_budget_max=param_budget_max,
        )
        if not presets:
            raise ValueError("No valid architecture presets fall inside the configured parameter budget.")
        preset = trial.suggest_categorical("architecture_preset", presets)
        hp.update(_decode_architecture_preset(preset))

    for name, spec in search_space.items():
        if use_architecture_presets and name in ARCHITECTURE_KEYS:
            continue
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
        if (hp["d_model"] // hp["n_heads"]) % 2 != 0:
            raise optuna.TrialPruned()

    # Derive d_ff from d_model (SwiGLU convention: ~2.67x, rounded to nearest 2)
    if "d_model" in hp and "d_ff" not in hp:
        hp["d_ff"] = _derive_d_ff(hp["d_model"])

    if (
        param_budget_min is not None
        and param_budget_max is not None
        and {"d_model", "n_layers", "n_heads", "d_ff"}.issubset(hp)
    ):
        vocab_size = int((model_defaults or {}).get("vocab_size", 50258))
        estimated_params = estimate_model_params(
            vocab_size=vocab_size,
            d_model=int(hp["d_model"]),
            n_layers=int(hp["n_layers"]),
            n_heads=int(hp["n_heads"]),
            d_ff=int(hp["d_ff"]),
        )
        if not param_budget_min <= estimated_params <= param_budget_max:
            raise optuna.TrialPruned()
        hp["estimated_params"] = estimated_params

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


def _build_base_config(project_config: ProjectConfig) -> dict:
    """Extract the model/training/logging sections for trial config rebuilding."""
    base_config: dict = {}
    if project_config.model:
        base_config["model"] = project_config.model.model_dump(mode="python")
    if project_config.training:
        base_config["training"] = project_config.training.model_dump(mode="python")
    if project_config.logging:
        base_config["logging"] = project_config.logging.model_dump(mode="python")
    return base_config


def _record_trial_metadata(
    trial: optuna.Trial,
    hp: dict,
    *,
    vocab_size: int,
) -> int | None:
    """Store architecture-related metadata on the Optuna trial."""
    resolved_arch = {k: hp[k] for k in ARCHITECTURE_KEYS if k in hp}
    if resolved_arch:
        trial.set_user_attr("model_hparams", resolved_arch)

    estimated_params = hp.get("estimated_params")
    if estimated_params is None and resolved_arch:
        estimated_params = estimate_model_params(
            vocab_size=vocab_size,
            d_model=resolved_arch["d_model"],
            n_layers=resolved_arch["n_layers"],
            n_heads=resolved_arch["n_heads"],
            d_ff=resolved_arch["d_ff"],
        )
    if estimated_params is not None:
        trial.set_user_attr("estimated_params", estimated_params)
    return estimated_params


def _log_trial_start(
    trial_number: int,
    trial_config: ProjectConfig,
    hp: dict,
    *,
    estimated_params: int | None = None,
) -> None:
    """Emit a compact summary for the sampled trial."""
    mc = trial_config.model
    lr = hp.get("learning_rate")
    lr_str = f"{lr:.2e}" if lr is not None else "?"
    params_str = f"{estimated_params:,}" if estimated_params is not None else "?"
    log.info(
        f"Trial {trial_number}: lr={lr_str}, d_model={mc.d_model}, "
        f"n_layers={mc.n_layers}, n_heads={mc.n_heads}, params={params_str}, "
        f"dropout={hp.get('dropout', '?')}"
    )


def _should_auto_relaunch_distributed_tune(*, export_only: bool) -> bool:
    """Return True when tune should self-relaunch under torchrun."""
    if export_only:
        return False
    if os.environ.get("PARROTLLM_DISABLE_AUTO_TORCHRUN") == "1":
        return False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return False
    if "LOCAL_RANK" in os.environ:
        return False

    import torch

    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def _relaunch_tune_with_torchrun() -> None:
    """Re-exec the current tune command under torchrun on a multi-GPU node."""
    import torch

    nproc_per_node = torch.cuda.device_count()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        *sys.argv,
    ]
    log.info(
        "Detected %d visible GPUs for tuning; relaunching under torchrun so each trial uses DDP.",
        nproc_per_node,
    )
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


# ── Objective ─────────────────────────────────────────────────────────────────

def _make_objective(tune_cfg: TuneConfig, base_config: dict):
    """Create the Optuna objective function."""
    # Convert search space to dicts for sample_hyperparams
    search_space = {
        name: spec.model_dump() for name, spec in tune_cfg.search_space.items()
    }

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparams(
            trial,
            search_space,
            model_defaults=base_config.get("model"),
            param_budget_min=tune_cfg.param_budget_min,
            param_budget_max=tune_cfg.param_budget_max,
        )
        project_config = build_trial_config(base_config, hp)

        device = get_device(project_config.training.device)
        mc = project_config.model

        estimated_params = _record_trial_metadata(
            trial,
            hp,
            vocab_size=base_config["model"]["vocab_size"],
        )
        _log_trial_start(trial.number, project_config, hp, estimated_params=estimated_params)

        from src.training.trainer import run_train
        import torch

        try:
            best_ppl = run_train(
                project_config,
                device=device,
                trial=trial,
            )
        except optuna.TrialPruned:
            log.info(f"Trial {trial.number} pruned.")
            raise
        except (RuntimeError, torch.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                log.warning(
                    f"Trial {trial.number} OOM — skipping. "
                    f"(d_model={mc.d_model}, n_layers={mc.n_layers}, n_heads={mc.n_heads})"
                )
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
    hp = _resolve_trial_params(best)

    # Derive d_ff and min_lr
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
        best_params = _resolve_trial_params(study.best_trial)
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")

        print(f"\nTop 5 trials:")
        top5 = sorted(completed, key=lambda t: t.value)[:5]
        for t in top5:
            params = _resolve_trial_params(t)
            lr = params.get("learning_rate", 0)
            wd = params.get("weight_decay", "?")
            dp = params.get("dropout", "?")
            d_model = params.get("d_model", "?")
            n_layers = params.get("n_layers", "?")
            n_heads = params.get("n_heads", "?")
            n_params = t.user_attrs.get("estimated_params", "?")
            if isinstance(n_params, int):
                params_str = f"{n_params / 1e6:.2f}M"
            else:
                params_str = str(n_params)
            print(
                f"  #{t.number:>3d}: ppl={t.value:>8.2f} | lr={lr:.2e} | "
                f"d={d_model} | L={n_layers} | H={n_heads} | params={params_str} | "
                f"wd={wd} | dropout={dp}"
            )

    print("=" * 70)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tune(
    project_config: ProjectConfig,
    *,
    n_trials_override: int | None = None,
    timeout_override: int | None = None,
    export_only: bool = False,
) -> None:
    """Run Optuna HP tuning using the project configuration.

    When launched via ``torchrun --nproc_per_node=N``, only rank 0 drives the
    Optuna study.  Non-zero ranks wait in a loop for rank 0 to broadcast trial
    configs, then participate in DDP training and report back the result.
    """
    if _should_auto_relaunch_distributed_tune(export_only=export_only):
        _relaunch_tune_with_torchrun()

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1

    # Initialise the process group once so all ranks can communicate.
    if use_ddp and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    tune_cfg = project_config.tune
    if rank == 0:
        study = create_study(tune_cfg)

    if export_only and rank == 0:
        if len(study.trials) == 0:
            print("No trials found. Run tuning first.")
        else:
            print_summary(study)
            export_best_params(study)
        if use_ddp:
            dist.barrier()
        return
    elif export_only:
        if use_ddp:
            dist.barrier()
        return

    n_trials = n_trials_override or tune_cfg.n_trials
    timeout = timeout_override or tune_cfg.timeout

    base_config = _build_base_config(project_config)

    if not use_ddp:
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
        return

    search_space = {
        name: spec.model_dump() for name, spec in tune_cfg.search_space.items()
    }

    if rank == 0:
        log.info(f"Starting Optuna study '{tune_cfg.name}' with {n_trials} trials (DDP, {world_size} GPUs)")

    trials_processed = 0
    tune_start_time = time.monotonic()

    while True:
        if rank == 0:
            timed_out = timeout is not None and (time.monotonic() - tune_start_time) >= timeout
            if timed_out or trials_processed >= n_trials:
                if timed_out:
                    log.info(
                        "Optuna timeout reached after %d processed trial(s); stopping DDP tuning.",
                        trials_processed,
                    )
                command = {"command": "stop"}
                trial = None
            else:
                trial = study.ask()
                try:
                    hp = sample_hyperparams(
                        trial,
                        search_space,
                        model_defaults=base_config.get("model"),
                        param_budget_min=tune_cfg.param_budget_min,
                        param_budget_max=tune_cfg.param_budget_max,
                    )
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    log.info(f"Trial {trial.number} pruned before launch.")
                    command = {"command": "skip"}
                    trial = None
                else:
                    trial_config = build_trial_config(base_config, hp)
                    estimated_params = _record_trial_metadata(
                        trial,
                        hp,
                        vocab_size=base_config["model"]["vocab_size"],
                    )
                    _log_trial_start(
                        trial.number,
                        trial_config,
                        hp,
                        estimated_params=estimated_params,
                    )
                    command = {
                        "command": "train",
                        "trial_number": trial.number,
                        "trial_config": trial_config.model_dump(mode="python"),
                    }
        else:
            command = None
            trial = None

        obj_list = [command]
        dist.broadcast_object_list(obj_list, src=0)
        command = obj_list[0]
        action = command["command"]

        if action == "stop":
            break

        trials_processed += 1
        if action == "skip":
            continue

        trial_config_dict = command["trial_config"]

        trial_config = ProjectConfig.model_validate(trial_config_dict)
        device = get_device(trial_config.training.device)

        from src.training.trainer import run_train

        try:
            trial_for_rank = trial if rank == 0 else None
            best_ppl = run_train(trial_config, device=device, trial=trial_for_rank)
        except optuna.TrialPruned:
            if rank == 0 and trial is not None:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                log.info(f"Trial {trial.number} pruned.")
            continue
        except (RuntimeError, torch.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                log.warning(f"Trial {command['trial_number']} OOM — skipping.")
                torch.cuda.empty_cache()
                best_ppl = float("inf")
            else:
                raise

        # Rank 0 reports back to Optuna.
        if rank == 0 and trial is not None:
            if best_ppl == float("inf"):
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, best_ppl)
                log.info(f"Trial {trial.number} finished: ppl={best_ppl:.2f}")

    if rank == 0:
        print_summary(study)
        export_best_params(study)
