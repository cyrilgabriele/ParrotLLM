from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.scripts import upload_run_to_hf


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_C_8B_CONFIG = REPO_ROOT / "configs" / "big_run" / "exp_c_8b.yaml"


def _make_run_dir(root: Path, name: str = "run_202660410_044337") -> Path:
    run_dir = root / "runs" / "big_run" / "exp_c_8b" / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("{}\n")
    return run_dir


def test_upload_uses_configured_repo_and_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upload_run_to_hf, "maybe_load_hf_token", lambda: "hf-test-token")

    calls: list[dict[str, object]] = []

    def fake_upload_run_dir_to_hub(run_dir_arg, upload_config, **kwargs):
        calls.append(
            {
                "run_dir": run_dir_arg,
                "repo_id": upload_config.repo_id,
                "repo_type": upload_config.repo_type,
                "token": kwargs["token"],
                "project_root": kwargs["project_root"],
            }
        )
        return {
            "repo_id": upload_config.repo_id,
            "repo_type": upload_config.repo_type,
            "path_in_repo": "runs/big_run/exp_c_8b/run_202660410_044337",
            "commit_url": "https://huggingface.co/datasets/ParrotLabs/Preprocessed/commit/abc",
            "commit_oid": "abc",
        }

    monkeypatch.setattr(
        upload_run_to_hf, "upload_run_dir_to_hub", fake_upload_run_dir_to_hub
    )

    result = upload_run_to_hf.main(
        [
            "--config",
            str(EXP_C_8B_CONFIG),
            "--run-dir",
            str(run_dir.relative_to(tmp_path)),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert calls == [
        {
            "run_dir": "runs/big_run/exp_c_8b/run_202660410_044337",
            "repo_id": "ParrotLabs/Preprocessed",
            "repo_type": "dataset",
            "token": "hf-test-token",
            "project_root": tmp_path,
        }
    ]
    assert "Destination: runs/big_run/exp_c_8b/run_202660410_044337" in captured.out
    assert "Commit URL: https://huggingface.co/datasets/ParrotLabs/Preprocessed/commit/abc" in captured.out


def test_dry_run_does_not_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fail_upload(*_args, **_kwargs):
        raise AssertionError("upload should not be called during dry-run")

    monkeypatch.setattr(upload_run_to_hf, "upload_run_dir_to_hub", fail_upload)

    result = upload_run_to_hf.main(
        [
            "--config",
            str(EXP_C_8B_CONFIG),
            "--run-dir",
            str(run_dir.relative_to(tmp_path)),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Repo: ParrotLabs/Preprocessed" in captured.out
    assert "Repo type: dataset" in captured.out
    assert "Dry run: no upload performed." in captured.out


def test_missing_run_directory_exits_with_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        upload_run_to_hf.main(
            [
                "--config",
                str(EXP_C_8B_CONFIG),
                "--run-dir",
                "runs/big_run/exp_c_8b/does_not_exist",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Run directory not found: runs/big_run/exp_c_8b/does_not_exist" in captured.err


def test_missing_hf_upload_exits_with_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run_dir(tmp_path)
    config_payload = yaml.safe_load(EXP_C_8B_CONFIG.read_text())
    config_payload["training"].pop("hf_upload")
    config_path = tmp_path / "without_hf_upload.yaml"
    config_path.write_text(yaml.safe_dump(config_payload))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        upload_run_to_hf.main(
            [
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir.relative_to(tmp_path)),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert f"Config {config_path} does not define training.hf_upload." in captured.err


def test_repo_name_prefixed_run_path_is_normalized_when_running_from_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "ParrotLLM"
    run_dir = _make_run_dir(repo_root)
    monkeypatch.chdir(repo_root)

    normalized = upload_run_to_hf._normalise_run_dir(
        Path("ParrotLLM") / run_dir.relative_to(repo_root)
    )

    assert normalized == run_dir
