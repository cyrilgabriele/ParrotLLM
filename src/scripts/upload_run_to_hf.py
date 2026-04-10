"""Upload an existing ParrotLLM run directory to the configured Hugging Face repo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from configs import load_project_config
from src.training.trainer import resolve_hf_repo_run_path, upload_run_dir_to_hub
from src.utils import maybe_load_hf_token


def _normalise_run_dir(raw_run_dir: str | Path, *, cwd: Path | None = None) -> Path:
    """Resolve the repo-name-prefixed path shape often copied from terminal trees."""

    cwd = Path.cwd() if cwd is None else cwd
    run_dir = Path(raw_run_dir)
    if run_dir.exists():
        return run_dir

    if not run_dir.is_absolute() and run_dir.parts and run_dir.parts[0] == cwd.name:
        candidate = cwd.joinpath(*run_dir.parts[1:])
        if candidate.exists():
            return candidate

    return run_dir


def _require_existing_run_dir(raw_run_dir: str | Path, *, cwd: Path | None = None) -> Path:
    run_dir = _normalise_run_dir(raw_run_dir, cwd=cwd)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {raw_run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload an existing ParrotLLM run directory to Hugging Face."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/big_run/exp_c_8b.yaml"),
        help="Project config containing training.hf_upload.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing local run directory to upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved Hub destination without uploading.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        project_config = load_project_config(args.config)
        training_config = project_config.training
        if training_config is None or training_config.hf_upload is None:
            raise ValueError(
                f"Config {args.config} does not define training.hf_upload."
            )

        run_dir = _require_existing_run_dir(args.run_dir)
        path_in_repo = resolve_hf_repo_run_path(
            str(run_dir),
            repo_prefix=training_config.hf_upload.path_in_repo,
            project_root=Path.cwd(),
        )

        print(f"Repo: {training_config.hf_upload.repo_id}")
        print(f"Repo type: {training_config.hf_upload.repo_type}")
        print(f"Source: {run_dir}")
        print(f"Destination: {path_in_repo}")

        if args.dry_run:
            print("Dry run: no upload performed.")
            return 0

        upload_result = upload_run_dir_to_hub(
            str(run_dir),
            training_config.hf_upload,
            token=maybe_load_hf_token(),
            project_root=Path.cwd(),
        )
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    print("Upload complete.")
    print(f"Repo: {upload_result['repo_id']}")
    print(f"Repo type: {upload_result['repo_type']}")
    print(f"Destination: {upload_result['path_in_repo']}")
    commit_url = upload_result.get("commit_url")
    commit_oid = upload_result.get("commit_oid")
    if commit_url:
        print(f"Commit URL: {commit_url}")
    if commit_oid:
        print(f"Commit OID: {commit_oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
