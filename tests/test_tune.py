"""Tests for Optuna HP search (src/training/tune.py)."""

import pytest


def test_load_tune_config():
    """tune.yaml loads and has required sections."""
    from tune import load_tune_config
    cfg = load_tune_config("configs/tuning/tune.yaml")
    assert "tune" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["tune"]["name"] == "parrotllm-hp-search"


def test_tune_config_validates():
    """tune.yaml validates through ProjectConfig."""
    from configs import load_project_config
    pc = load_project_config("configs/tuning/tune.yaml")
    assert pc.tune is not None
    assert pc.tune.name == "parrotllm-hp-search"
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
        assert hp["batch_size"] in [8, 16, 32, 64]
        assert hp["d_model"] % hp["n_heads"] == 0
        assert "d_ff" in hp
    assert valid_count > 0, "All 20 trials were pruned — search space may be too constrained"


def test_invalid_d_model_n_heads_prunes():
    """Invalid d_model/n_heads combinations raise TrialPruned."""
    import optuna
    from src.training.tune import sample_hyperparams

    search_space = {
        "d_model": {"type": "categorical", "choices": [192]},
        "n_heads": {"type": "categorical", "choices": [8]},
    }
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    hp = sample_hyperparams(trial, search_space)
    assert hp["d_model"] % hp["n_heads"] == 0


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
    assert project_config.model.d_model == hp["d_model"]
