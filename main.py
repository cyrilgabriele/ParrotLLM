"""Project entry point that enforces a single configuration source."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import load_project_config, load_project_config_from_checkpoint
from src.logging_utils import init_logging
from src.utils import get_device, set_seed, maybe_load_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="ParrotLLM")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "train", "tune", "eval", "inference", "chat", "dashboard", "sft", "dpo"],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume training from the checkpoint passed via --checkpoint.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--mock-testing", action="store_true", default=None)
    # dashboard-specific
    parser.add_argument("--open", action="store_true",
                        help="Open browser automatically when starting the dashboard")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share URL")
    parser.add_argument("--tui", action="store_true",
                        help="Use terminal UI instead of Gradio")
    parser.add_argument("--tui-refresh", type=int, default=2, metavar="N",
                        help="TUI refresh interval in seconds (default: 2)")
    parser.add_argument("--tui-run", default=None, metavar="RUN_NAME",
                        help="Pin TUI to a specific run, e.g. run_20260406_130146 (default: latest)")
    # tune-specific
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of Optuna trials")
    parser.add_argument("--timeout-tune", type=int, default=None,
                        help="Override Optuna timeout (seconds)")
    parser.add_argument("--export-only", action="store_true",
                        help="Just export best params from existing study")

    args = parser.parse_args()

    resume_checkpoint = _resolve_train_checkpoint(args, parser)
    project_config = _load_effective_project_config(args, resume_checkpoint)

    logging_cfg = project_config.logging
    if logging_cfg:
        init_logging(
            console_level=logging_cfg.console_level,
            component_levels=logging_cfg.components if logging_cfg.components else None,
        )
    else:
        init_logging()

    project_config_payload = project_config.model_dump(mode="python")
    HF_TOKEN = maybe_load_hf_token()

    SEED = 42
    set_seed(SEED)

    if args.stage == "preprocess":
        preprocess_cfg = _require_section(project_config.preprocess, "preprocess")
        from src.data.preprocess import run_preprocess

        run_preprocess(preprocess_cfg, SEED)
        return

    if args.stage == "tune":
        _require_section(project_config.tune, "tune")
        _require_section(project_config.model, "model")
        _require_section(project_config.training, "training")
        from src.training.tune import run_tune

        run_tune(
            project_config,
            n_trials_override=args.n_trials,
            timeout_override=args.timeout_tune,
            export_only=args.export_only,
        )
        return

    if args.stage == "train":
        training_cfg = _require_section(project_config.training, "training")
        _require_section(project_config.model, "model")

        device = get_device(training_cfg.device)
        from src.training.trainer import run_train

        run_train(project_config, device=device, checkpoint=resume_checkpoint)
        return

    if args.stage == "eval":
        eval_cfg = _require_section(project_config.eval, "eval")
        checkpoint_path = _require_checkpoint(args.checkpoint, stage="eval")
        device = get_device(eval_cfg.device)
        from src.eval.perplexity import run_eval

        run_eval(
            project_config,
            project_config_payload,
            checkpoint=checkpoint_path,
            device=device,
            hf_token=HF_TOKEN
        )
        return

    if args.stage == "inference":
        inference_cfg = _require_section(project_config.inference, "inference")
        checkpoint_path = args.checkpoint
        if not args.mock_testing:
            checkpoint_path = _require_checkpoint(args.checkpoint, stage="inference")
        device = get_device(inference_cfg.device)
        from src.eval.inference import run_inference

        run_inference(
            project_config,
            checkpoint=checkpoint_path,
            device=device,
            prompt=args.prompt,
            max_tokens_override=args.max_tokens,
            temperature_override=args.temperature,
            leaderboard=args.leaderboard,
            mock_testing=args.mock_testing,
            hf_token=HF_TOKEN
        )
        return

    if args.stage == "sft":
        # VL07 — Supervised Fine-Tuning stage. Requires:
        #   (i)   an `sft:` block in the YAML config (see configs/default.yaml),
        #   (ii)  a base pretraining checkpoint, sourced from --checkpoint or
        #         from sft.base_checkpoint in the YAML (CLI takes precedence).
        sft_cfg = _require_section(project_config.sft, "sft")
        _require_section(project_config.model, "model")
        checkpoint_path = _resolve_sft_checkpoint(args.checkpoint, sft_cfg.base_checkpoint)
        device = get_device(sft_cfg.device)
        from src.post_training.sft.trainer import run_sft

        run_sft(project_config, checkpoint=checkpoint_path, device=device)
        return

    if args.stage == "dpo":
        # VL08 — Direct Preference Optimization. Requires:
        #   (i)   a `dpo:` block in the YAML config,
        #   (ii)  an SFT checkpoint passed via --checkpoint (used as BOTH
        #         the policy initialisation AND the frozen reference model).
        dpo_cfg = _require_section(project_config.dpo, "dpo")
        _require_section(project_config.model, "model")
        checkpoint_path = _resolve_sft_checkpoint(args.checkpoint, dpo_cfg.base_checkpoint)
        device = get_device(dpo_cfg.device)
        from src.post_training.dpo.trainer import run_dpo

        run_dpo(project_config, checkpoint=checkpoint_path, device=device)
        return

    if args.stage == "chat":
        chat_cfg = _require_section(project_config.chat, "chat")
        device = get_device(chat_cfg.device)
        from src.chat.app import run_chat

        run_chat(project_config, device=device)

    if args.stage == "dashboard":
        from pathlib import Path as _Path
        training_cfg = project_config.training
        runs_dir = _Path(training_cfg.runs_dir) if training_cfg else _Path("runs")

        if args.tui:
            from src.dashboard.tui import run_tui
            run_tui(runs_dir=runs_dir, refresh=args.tui_refresh, run_name=args.tui_run)
        else:
            from src.dashboard.app import run_dashboard
            run_dashboard(
                runs_dir=runs_dir,
                config_path=args.config,
                share=args.share,
                open_browser=args.open,
            )
        return


def _require_section(value, name: str):
    if value is None:
        raise ValueError(f"Configuration section '{name}' is missing from the YAML file.")
    return value


def _resolve_train_checkpoint(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    if args.stage != "train":
        return None
    if args.resume_training:
        return _require_checkpoint(args.checkpoint, stage="train")
    if args.checkpoint:
        parser.error(
            "--checkpoint only resumes training when used together with --resume-training."
        )
    return None


def _load_effective_project_config(
    args: argparse.Namespace,
    resume_checkpoint: str | None,
):
    if args.stage == "train" and resume_checkpoint:
        return load_project_config_from_checkpoint(resume_checkpoint)
    return load_project_config(args.config)


def _require_checkpoint(path: str | None, stage: str) -> str:
    if not path:
        raise ValueError(f"--checkpoint is required for stage '{stage}'.")
    return path


def _resolve_sft_checkpoint(cli_path: str | None, yaml_default: str | None) -> str:
    """Choose the SFT base checkpoint. CLI flag wins; YAML is the fallback.

    SFT on random weights is meaningless (VL07 slide 12), so one of the
    two sources must be set. Empty strings from argparse fall through.
    """
    path = (cli_path or "").strip() or (yaml_default or "").strip()
    if not path:
        raise ValueError(
            "SFT requires a base pretraining checkpoint. Pass --checkpoint "
            "or set sft.base_checkpoint in the YAML."
        )
    return path


if __name__ == "__main__":
    main()
