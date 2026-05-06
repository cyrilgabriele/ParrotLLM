from __future__ import annotations

from pathlib import Path

from src.post_training import hf_cache
from src.post_training.hf_cache import cleanup_hf_dataset_cache


def test_cleanup_hf_dataset_cache_removes_explicit_cache_dir(tmp_path):
    cache_dir = tmp_path / "datasets-cache"
    cache_dir.mkdir()
    (cache_dir / "data.arrow").write_text("cached", encoding="utf-8")

    removed = cleanup_hf_dataset_cache(
        cache_dir=cache_dir,
        include_hub_dataset_repos=False,
    )

    assert removed == [cache_dir]
    assert not cache_dir.exists()


def test_cleanup_hf_dataset_cache_only_removes_dataset_hub_repos(
    tmp_path,
    monkeypatch,
):
    hub_dir = tmp_path / "hub"
    dataset_repo = hub_dir / "datasets--allenai--sciq"
    model_repo = hub_dir / "models--openai-community--gpt2"
    dataset_repo.mkdir(parents=True)
    model_repo.mkdir(parents=True)
    (dataset_repo / "blob").write_text("dataset", encoding="utf-8")
    (model_repo / "blob").write_text("model", encoding="utf-8")

    monkeypatch.setattr(hf_cache, "_default_datasets_cache", lambda: None)
    monkeypatch.setattr(hf_cache, "_default_hub_cache", lambda: Path(hub_dir))

    removed = cleanup_hf_dataset_cache()

    assert dataset_repo in removed
    assert not dataset_repo.exists()
    assert model_repo.exists()
