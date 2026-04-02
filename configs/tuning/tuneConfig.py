"""Pydantic configuration for Optuna hyperparameter tuning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchParamConfig(BaseModel):
    """Single hyperparameter search space entry."""
    model_config = ConfigDict(extra="allow")

    type: Literal["log_uniform", "uniform", "int", "categorical"] = Field(...)
    low: float | None = None
    high: float | None = None
    step: int | None = None
    choices: list | None = None


class TuneConfig(BaseModel):
    """Validated configuration for Optuna HP tuning."""
    model_config = ConfigDict(extra="ignore")

    # Study settings
    name: str = Field(default="parrotllm-hp-search", description="Optuna study name.")
    storage: str = Field(
        default="optuna_studies/parrotllm.db",
        description="SQLite path for study persistence.",
    )
    n_trials: int = Field(default=50, ge=1, description="Number of trials to run.")
    timeout: int | None = Field(default=None, description="Max seconds for entire study.")
    sampler: Literal["tpe", "random"] = Field(default="tpe", description="Sampling algorithm.")
    pruner: Literal["hyperband", "median"] = Field(default="hyperband", description="Pruning algorithm.")
    pruner_kwargs: dict = Field(default_factory=dict, description="Keyword arguments for the pruner.")
    param_budget_min: int | None = Field(
        default=None,
        ge=1,
        description="Optional minimum model parameter budget for architecture search.",
    )
    param_budget_max: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum model parameter budget for architecture search.",
    )

    # Search space
    search_space: dict[str, SearchParamConfig] = Field(
        ..., description="Hyperparameter search space definitions.",
    )
