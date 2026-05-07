"""Project entry point that enforces a single configuration source."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _delegate_leaderboard_inference_if_requested() -> None:
    """Route `python main.py --stage inference --leaderboard ...` to the
    polished submission entrypoint at Submissions/parrotlabs_parrotllm/main.py.

    The factsheet §4.4 harness clones our repo and invokes the literal
    contract `python main.py --stage inference --checkpoint ... --device auto
    --leaderboard --seed 0`. The repo-root inference path
    (`src/eval/inference.py`) Alpaca-wraps every prompt and skips MC /
    LAMBADA shape dispatch + cloze-scoring, so it would emit malformed
    answers for the public benchmarks. Delegating to the submission keeps
    one source of truth for leaderboard inference. Detection is done
    pre-argparse so unrecognised flags from the contract (e.g. --device,
    which the rest of this script doesn't use) don't error out.
    """
    argv = sys.argv
    if "--stage" not in argv or "--leaderboard" not in argv:
        return
    try:
        stage_val = argv[argv.index("--stage") + 1]
    except (IndexError, ValueError):
        return
    if stage_val != "inference":
        return

    submission_main = (
        Path(__file__).resolve().parent
        / "Submissions" / "parrotlabs_parrotllm" / "main.py"
    )
    if not submission_main.is_file():
        return

    forwarded = list(argv[1:])
    # Resolve --checkpoint against the *current* cwd before exec. os.execv
    # preserves cwd, so relative paths would still work — but freezing the
    # absolute path here keeps the submission's error messages pointing at
    # what the TA actually typed, regardless of where the submission script
    # internally re-resolves it.
    if "--checkpoint" in forwarded:
        idx = forwarded.index("--checkpoint")
        if idx + 1 < len(forwarded):
            ckpt = Path(forwarded[idx + 1])
            if not ckpt.is_absolute():
                forwarded[idx + 1] = str((Path.cwd() / ckpt).resolve())

    os.execv(sys.executable, [sys.executable, str(submission_main), *forwarded])


_delegate_leaderboard_inference_if_requested()

from configs import load_project_config, load_project_config_from_checkpoint
from src.logging_utils import init_logging
from src.utils import get_device, set_seed, maybe_load_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="ParrotLLM")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "preprocess",
            "preprocess-streaming",
            "train",
            "tune",
            "eval",
            "inference",
            "chat",
            "dashboard",
            "sft-download",
            "sft-prepare",
            "sft",
            "benchmark",
            "dpo-prepare",
            "dpo",
        ],
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
    parser.add_argument("--seed", type=int, default=42)
    # Factsheet §4.4 inference contract. Non-leaderboard stages ignore this
    # (the YAML drives device selection); leaderboard inference is delegated
    # to the submission script before this parser even runs.
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (auto/cuda/mps/cpu). Required by the "
             "leaderboard inference contract (factsheet §4.4).",
    )
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
    # benchmark-specific
    parser.add_argument(
        "--tier",
        choices=["smoke", "quick", "full"],
        default="quick",
        help="Benchmark tier; smoke=5 items, quick=200 items, full=all.",
    )
    parser.add_argument(
        "--submission-name",
        default="ParrotLLM",
        help="Name of the submission folder under external/PikoGPT_Leaderboard/Submissions/.",
    )

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

    SEED = int(args.seed)
    set_seed(SEED)

    if args.stage == "preprocess":
        preprocess_cfg = _require_section(project_config.preprocess, "preprocess")
        from src.data.preprocess import run_preprocess

        run_preprocess(preprocess_cfg, SEED)
        return

    if args.stage == "preprocess-streaming":
        streaming_cfg = _require_section(
            project_config.streaming_preprocess, "streaming_preprocess"
        )
        from src.data.preprocess_streaming import run_preprocess_streaming

        run_preprocess_streaming(streaming_cfg)
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
        device = get_device(args.device or inference_cfg.device)
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

    if args.stage == "chat":
        chat_cfg = _require_section(project_config.chat, "chat")
        device = get_device(chat_cfg.device)
        from src.chat.app import run_chat

        run_chat(project_config, device=device)
        return

    if args.stage == "benchmark":
        import subprocess
        from src.posttraining.benchmarks.harness import BenchmarkRunSpec, run_benchmark
        from src.posttraining.benchmarks.compare import build_comparison_markdown

        checkpoint_path = _require_checkpoint(args.checkpoint, stage="benchmark")
        leaderboard_repo = Path("external/PikoGPT_Leaderboard")
        registry_dir = Path("runs/benchmarks")
        external_yaml = registry_dir / "external_groups.yaml"

        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()

        spec = BenchmarkRunSpec(
            checkpoint=Path(checkpoint_path).resolve(),
            tier=args.tier,
            submission_name=args.submission_name,
            leaderboard_repo=leaderboard_repo,
            registry_dir=registry_dir,
            git_sha=git_sha,
        )
        result = run_benchmark(spec)
        print(f"\nBenchmark result for {result.checkpoint_basename} @ tier={result.tier}:")
        print(f"  hellaswag={result.scores.get('hellaswag', 0):.2f}")
        print(f"  openbookqa={result.scores.get('openbookqa', 0):.2f}")
        print(f"  winogrande={result.scores.get('winogrande', 0):.2f}")
        print(f"  lambada={result.scores.get('lambada', 0):.2f}")
        print(f"  PII (named) = {result.pii_named:.2f}")
        print(f"  wall-clock = {result.wall_clock_seconds:.1f}s")
        print()
        print(build_comparison_markdown(registry_dir, external_yaml))
        return

    if args.stage == "sft-download":
        _require_section(project_config.sft, "sft")
        from src.posttraining.download import run_download_sft

        run_download_sft(project_config, hf_token=HF_TOKEN)
        return

    if args.stage == "sft-prepare":
        _require_section(project_config.sft, "sft")
        from src.posttraining.prepare import run_prepare_sft

        run_prepare_sft(project_config, seed=SEED, hf_token=HF_TOKEN)
        return

    if args.stage == "dpo-prepare":
        _require_section(project_config.dpo, "dpo")
        from src.posttraining.dpo.prepare import run_prepare_dpo

        run_prepare_dpo(project_config, seed=SEED, hf_token=HF_TOKEN)
        return

    if args.stage == "dpo":
        dpo_cfg = _require_section(project_config.dpo, "dpo")
        _require_section(project_config.model, "model")
        device = get_device(dpo_cfg.device)
        from src.posttraining.dpo.trainer import run_dpo

        run_dpo(
            project_config,
            device=device,
            checkpoint=args.checkpoint,
        )
        return

    if args.stage == "sft":
        sft_cfg = _require_section(project_config.sft, "sft")
        _require_section(project_config.model, "model")
        device = get_device(sft_cfg.device)
        from src.posttraining.trainer import run_sft

        run_sft(
            project_config,
            device=device,
            checkpoint=args.checkpoint,
        )
        return

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
    p = Path(path)
    if p.is_dir():
        return _find_best_checkpoint(p, stage)
    return path


def _find_best_checkpoint(checkpoint_dir: Path, stage: str) -> str:
    """Auto-select the best checkpoint from a directory (lowest val loss)."""
    best_files = sorted(checkpoint_dir.glob("best_loss_*.pt"))
    if not best_files:
        raise ValueError(
            f"No best_loss_*.pt checkpoints found in '{checkpoint_dir}' for stage '{stage}'."
        )
    # Filenames: best_loss_3p1234_epoch_0001_step_0003052.pt
    # Parse loss value (e.g. "3p1234" -> 3.1234) and pick lowest.
    def _parse_loss(f: Path) -> float:
        name = f.stem  # best_loss_3p1234_epoch_0001_step_0003052
        loss_str = name.split("best_loss_")[1].split("_epoch_")[0]
        return float(loss_str.replace("p", "."))

    best = min(best_files, key=_parse_loss)
    print(f"[eval] Auto-selected best checkpoint: {best.name}")
    return str(best)


if __name__ == "__main__":
    main()
