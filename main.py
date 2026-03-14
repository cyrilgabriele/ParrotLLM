"""Project entry point that enforces a single configuration source."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import load_project_config
from src.logging_utils import init_logging
from src.utils import get_device, set_seed, maybe_load_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="ParrotLLM")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "train", "eval", "inference", "chat"],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--mock-testing", action="store_true", default=None)

    args = parser.parse_args()

    project_config = load_project_config(args.config)

    logging_cfg = project_config.logging
    if logging_cfg:
        init_logging(
            console_level=logging_cfg.console_level,
            component_levels=logging_cfg.components if logging_cfg.components else None,
        )
    else:
        init_logging()

    config_dict = project_config.model_dump(mode="python")
    HF_TOKEN = maybe_load_hf_token()

    SEED = 42
    set_seed(SEED)

    if args.stage == "preprocess":
        preprocess_cfg = _require_section(project_config.preprocess, "preprocess")
        from src.data.preprocess import run_preprocess

        run_preprocess(preprocess_cfg, SEED)
        return

    if args.stage == "train":
        training_cfg = _require_section(project_config.training, "training")
        _require_section(project_config.model, "model")
        device = get_device(training_cfg.device)
        from src.training.trainer import run_train

        run_train(project_config, config_dict, device=device, checkpoint=args.checkpoint)
        return

    if args.stage == "eval":
        eval_cfg = _require_section(project_config.eval, "eval")
        checkpoint_path = _require_checkpoint(args.checkpoint, stage="eval")
        device = get_device(eval_cfg.device)
        from src.eval.perplexity import run_eval

        run_eval(
            project_config,
            config_dict,
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

    if args.stage == "chat":
        chat_cfg = _require_section(project_config.chat, "chat")
        device = get_device(chat_cfg.device)
        from src.chat.app import run_chat

        run_chat(project_config, device=device)


def _require_section(value, name: str):
    if value is None:
        raise ValueError(f"Configuration section '{name}' is missing from the YAML file.")
    return value


def _require_checkpoint(path: str | None, stage: str) -> str:
    if not path:
        raise ValueError(f"--checkpoint is required for stage '{stage}'.")
    return path


if __name__ == "__main__":
    main()
