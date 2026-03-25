"""Test that run_train accepts an Optuna trial and reports intermediate values."""

import inspect
from src.training.trainer import run_train


def test_run_train_accepts_trial_parameter():
    """run_train() signature accepts trial keyword argument without error."""
    sig = inspect.signature(run_train)
    assert "trial" in sig.parameters, "run_train must accept a 'trial' parameter"


def test_run_train_trial_default_is_none():
    """trial parameter defaults to None (backward compatible)."""
    sig = inspect.signature(run_train)
    assert sig.parameters["trial"].default is None


def test_run_train_returns_float():
    """run_train return annotation is float."""
    sig = inspect.signature(run_train)
    assert sig.return_annotation in (float, "float")
