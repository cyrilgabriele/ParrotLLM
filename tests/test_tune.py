"""Tests for tune.py — Optuna HP search."""

import yaml
import pytest


def test_load_tune_config():
    """tune.yaml loads and has required sections."""
    from tune import load_tune_config
    cfg = load_tune_config("configs/tune.yaml")
    assert "study" in cfg
    assert "base_config" in cfg
    assert "search_space" in cfg
    assert cfg["study"]["name"] == "parrotllm-hp-search"


def test_sample_hyperparams_returns_valid_config():
    """sample_hyperparams produces a valid ProjectConfig patch."""
    import optuna
    from tune import load_tune_config, sample_hyperparams

    cfg = load_tune_config("configs/tune.yaml")
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    hp = sample_hyperparams(trial, cfg["search_space"])

    assert 1e-4 <= hp["learning_rate"] <= 3e-3
    assert hp["batch_size"] in [8, 16, 32, 64]
    assert hp["d_model"] % hp["n_heads"] == 0
    assert "d_ff" in hp  # derived


def test_build_trial_config():
    """build_trial_config merges sampled HPs into base config."""
    import optuna
    from tune import load_tune_config, sample_hyperparams, build_trial_config
    from configs import ProjectConfig

    cfg = load_tune_config("configs/tune.yaml")
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    hp = sample_hyperparams(trial, cfg["search_space"])
    project_config = build_trial_config(cfg["base_config"], hp)

    assert isinstance(project_config, ProjectConfig)
    assert project_config.training.learning_rate == hp["learning_rate"]
    assert project_config.model.d_model == hp["d_model"]
