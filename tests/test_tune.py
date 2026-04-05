"""Tests for Optuna HP search (src/training/tune.py)."""

import pytest


def test_load_tune_config():
    """tune.yaml loads and has required sections."""
    from tune import load_tune_config
    cfg = load_tune_config("configs/tuning/tune.yaml")
    assert "tune" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["tune"]["name"] == "parrotllm-definitive-local"


def test_tune_config_validates():
    """tune.yaml validates through ProjectConfig."""
    from configs import load_project_config
    pc = load_project_config("configs/tuning/tune.yaml")
    assert pc.tune is not None
    assert pc.tune.name == "parrotllm-definitive-local"
    assert pc.model is not None
    assert pc.training is not None
    assert len(pc.tune.search_space) > 0


def test_sample_hyperparams_returns_valid_config():
    """sample_hyperparams produces a valid ProjectConfig patch (or prunes invalid combos)."""
    import optuna
    from src.training.tune import sample_hyperparams
    from configs import load_project_config

    pc = load_project_config("configs/tuning/tune.yaml")
    search_space = {
        name: spec.model_dump() for name, spec in pc.tune.search_space.items()
    }

    study = optuna.create_study(direction="minimize")
    valid_count = 0
    for _ in range(20):
        trial = study.ask()
        try:
            hp = sample_hyperparams(trial, search_space)
        except optuna.TrialPruned:
            continue
        valid_count += 1
        assert 1e-4 <= hp["learning_rate"] <= 3e-3
        assert hp["batch_size"] in [32, 64]
        assert hp["gradient_accumulation_steps"] in [1, 2, 4]
        assert hp["lr_schedule"] in ["wsd", "cosine"]
        assert hp["min_lr"] == pytest.approx(hp["learning_rate"] * 0.1)
        assert "architecture_preset" not in hp
    assert valid_count > 0, "All 20 trials were pruned — search space may be too constrained"


def test_invalid_d_model_n_heads_prunes():
    """Invalid d_model/n_heads combinations raise TrialPruned."""
    import optuna
    from src.training.tune import sample_hyperparams

    search_space = {
        "d_model": {"type": "categorical", "choices": [120]},
        "n_heads": {"type": "categorical", "choices": [8]},
    }
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    with pytest.raises(optuna.TrialPruned):
        sample_hyperparams(trial, search_space)


def test_build_trial_config():
    """build_trial_config merges sampled HPs into base config."""
    import optuna
    from src.training.tune import sample_hyperparams, build_trial_config
    from configs import ProjectConfig, load_project_config

    pc = load_project_config("configs/tuning/tune.yaml")
    base_config = {
        "model": pc.model.model_dump(mode="python"),
        "training": pc.training.model_dump(mode="python"),
    }
    if pc.logging:
        base_config["logging"] = pc.logging.model_dump(mode="python")

    search_space = {
        name: spec.model_dump() for name, spec in pc.tune.search_space.items()
    }

    study = optuna.create_study(direction="minimize")
    for _ in range(20):
        trial = study.ask()
        try:
            hp = sample_hyperparams(trial, search_space)
            break
        except optuna.TrialPruned:
            continue
    else:
        pytest.skip("Could not get a valid trial in 20 attempts")

    project_config = build_trial_config(base_config, hp)

    assert isinstance(project_config, ProjectConfig)
    assert project_config.training.learning_rate == hp["learning_rate"]
    assert project_config.model.d_model == base_config["model"]["d_model"]
    assert project_config.model.dropout == hp["dropout"]


def test_sample_hyperparams_respects_parameter_budget():
    """Architecture presets stay inside the configured parameter window."""
    import optuna
    from src.training.tune import sample_hyperparams

    search_space = {
        "d_model": {"type": "int", "low": 104, "high": 144, "step": 8},
        "n_layers": {"type": "int", "low": 6, "high": 27, "step": 1},
        "n_heads": {"type": "categorical", "choices": [2, 4, 6, 8, 12, 16]},
        "d_ff": {"type": "int", "low": 272, "high": 384, "step": 16},
        "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 3e-3},
    }
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    hp = sample_hyperparams(
        trial,
        search_space,
        model_defaults={"vocab_size": 50258},
        param_budget_min=8_500_000,
        param_budget_max=8_750_000,
    )

    assert "architecture_preset" in trial.params
    assert 8_500_000 <= hp["estimated_params"] <= 8_750_000
    assert 272 <= hp["d_ff"] <= 384
    assert hp["d_model"] % hp["n_heads"] == 0
    assert (hp["d_model"] // hp["n_heads"]) % 2 == 0
