"""Backward-compatible entry point — prefer: python main.py --stage tune

Usage:
    python tune.py                          # uses configs/tune.yaml
    python tune.py --config configs/tune.yaml --n-trials 100
    python tune.py --resume                 # resume previous study
"""

from __future__ import annotations

import argparse

from configs import load_project_config
from src.logging_utils import init_logging
from src.training.tune import run_tune
from src.utils import set_seed


# Re-export for test compatibility
from src.training.tune import (  # noqa: F401
    sample_hyperparams,
    build_trial_config,
)


def load_tune_config(path: str) -> dict:
    """Load tune config — kept for backward compatibility with tests."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Validate it has the required sections
    for key in ("tune", "model", "training"):
        if key not in cfg:
            raise ValueError(f"tune config missing required section: '{key}'")
    return cfg


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

    project_config = load_project_config(args.config)

    run_tune(
        project_config,
        n_trials_override=args.n_trials,
        timeout_override=args.timeout,
        export_only=args.export_only,
    )


if __name__ == "__main__":
    main()
