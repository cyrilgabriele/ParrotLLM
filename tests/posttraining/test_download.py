from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from configs import ProjectConfig, SFTSourceConfig
from src.posttraining.download import run_download_sft
from src.posttraining.prepare import PreparedExample, _load_records, get_source_snapshot_path


class DummyTokenizer:
    eos_token_id = 999

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        verbose: bool = True,
    ):
        del add_special_tokens, verbose
        payload = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload


class FakeDataset(list):
    @property
    def num_rows(self) -> int:
        return len(self)


def _build_download_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "model": {
                "vocab_size": 1024,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 2,
                "d_ff": 64,
                "context_length": 128,
                "bias": False,
                "dropout": 0.0,
                "rope_theta": 10000.0,
                "gradient_checkpointing": False,
            },
            "sft": {
                "device": "cpu",
                "base_checkpoint": str(tmp_path / "base.pt"),
                "cache_dir": str(tmp_path / "hf_cache"),
                "raw_dir": str(tmp_path / "raw"),
                "prepared_dir": str(tmp_path / "prepared"),
                "runs_dir": str(tmp_path / "runs"),
                "system_prompt": "You are helpful.",
                "max_seq_length": 128,
                "sources": [
                    {
                        "name": "wildchat_gpt4",
                        "loader": "wildchat",
                        "path": "allenai/WildChat",
                        "target_examples": 5,
                        "min_turns": 2,
                        "max_turns": 4,
                    }
                ],
                "decontam_datasets": [
                    {
                        "name": "wikitext103_test",
                        "loader": "local_disk",
                        "path": str(tmp_path / "data" / "wikitext-103-test"),
                        "field": "text",
                    },
                    {
                        "name": "hellaswag",
                        "loader": "huggingface",
                        "path": "Rowan/hellaswag",
                        "field": "ctx",
                        "split": "validation",
                    },
                ],
            },
        }
    )


def test_load_records_uses_shared_cache_and_token(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_load_dataset(path, subset, **kwargs):
        captured["path"] = path
        captured["subset"] = subset
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("src.posttraining.prepare.load_dataset", fake_load_dataset)

    source_cfg = SFTSourceConfig.model_validate(
        {
            "name": "wildchat_gpt4",
            "loader": "wildchat",
            "path": "allenai/WildChat",
            "target_examples": 1,
        }
    )

    _load_records(source_cfg, cache_dir=tmp_path / "cache", hf_token="secret")

    assert captured["path"] == "allenai/WildChat"
    assert captured["kwargs"]["cache_dir"] == str(tmp_path / "cache")
    assert captured["kwargs"]["token"] == "secret"


def test_run_download_sft_prefetches_and_validates(monkeypatch, tmp_path: Path):
    cfg = _build_download_config(tmp_path)
    local_wiki_dir = tmp_path / "data" / "wikitext-103-test"

    def fake_download_wikitext():
        local_wiki_dir.mkdir(parents=True, exist_ok=True)
        (local_wiki_dir / "sample.txt").write_text("sample wiki text", encoding="utf-8")

    def fake_download_fasttext():
        return None

    def fake_load_dataset(path, subset=None, **kwargs):
        del subset, kwargs
        if path == "allenai/WildChat":
            return Dataset.from_list(
                [{"conversation": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}]
            )
        if path == "Rowan/hellaswag":
            return Dataset.from_list([{"ctx": "A context prompt"}])
        raise AssertionError(f"Unexpected dataset path: {path}")

    def fake_collect_candidates(*args, **kwargs):
        del args, kwargs
        return [
            PreparedExample(
                source="wildchat_gpt4",
                tags=["chat"],
                quality_score=1.0,
                prompt_hash="abc",
                prompt_text="Question",
                full_text_hash="def",
                tokens=[1, 2, 3],
                loss_mask=[0, 1, 0],
                messages=[
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                metadata={},
            )
        ]

    monkeypatch.setattr("src.posttraining.download.build_tokenizer", lambda: DummyTokenizer())
    monkeypatch.setattr("src.posttraining.download.download_wikitext103_test", fake_download_wikitext)
    monkeypatch.setattr("src.posttraining.download.download_fasttext_langdetect", fake_download_fasttext)
    monkeypatch.setattr("src.posttraining.download.load_dataset", fake_load_dataset)
    monkeypatch.setattr("src.posttraining.download._collect_candidates_for_source", fake_collect_candidates)

    manifest = run_download_sft(cfg, hf_token="token-123")

    manifest_path = Path(cfg.sft.prepared_dir) / "download_manifest.json"
    snapshot_path = get_source_snapshot_path(Path(cfg.sft.raw_dir), cfg.sft.sources[0])
    assert manifest_path.exists()
    assert manifest["cache_dir"] == str(Path(cfg.sft.cache_dir))
    assert manifest["cache_cleaned"] is True
    assert manifest["raw_dir"] == str(Path(cfg.sft.raw_dir))
    assert manifest["sources"][0]["preview_candidates"] == 1
    assert manifest["sources"][0]["snapshot_path"] == str(snapshot_path)
    assert manifest["decontam_datasets"][0]["sample_texts"] >= 1
    assert snapshot_path is not None and snapshot_path.exists()
    assert not Path(cfg.sft.cache_dir).exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sources"][0]["name"] == "wildchat_gpt4"


def test_run_download_sft_falls_back_to_raw_schema_validation(monkeypatch, tmp_path: Path):
    cfg = _build_download_config(tmp_path)
    local_wiki_dir = tmp_path / "data" / "wikitext-103-test"

    def fake_download_wikitext():
        local_wiki_dir.mkdir(parents=True, exist_ok=True)
        (local_wiki_dir / "sample.txt").write_text("sample wiki text", encoding="utf-8")

    def fake_load_dataset(path, subset=None, **kwargs):
        del subset, kwargs
        if path == "allenai/WildChat":
            return Dataset.from_list(
                [
                    {
                        "conversation": [
                            {"role": "user", "content": "Q"},
                            {"role": "assistant", "content": "A"},
                        ]
                    }
                ]
            )
        if path == "Rowan/hellaswag":
            return Dataset.from_list([{"ctx": "A context prompt"}])
        raise AssertionError(f"Unexpected dataset path: {path}")

    def fake_collect_candidates(*args, **kwargs):
        del args, kwargs
        return []

    monkeypatch.setattr("src.posttraining.download.build_tokenizer", lambda: DummyTokenizer())
    monkeypatch.setattr("src.posttraining.download.download_wikitext103_test", fake_download_wikitext)
    monkeypatch.setattr("src.posttraining.download.download_fasttext_langdetect", lambda: None)
    monkeypatch.setattr("src.posttraining.download.load_dataset", fake_load_dataset)
    monkeypatch.setattr("src.posttraining.download._collect_candidates_for_source", fake_collect_candidates)

    manifest = run_download_sft(cfg)

    assert manifest["sources"][0]["preview_candidates"] == 0
    assert manifest["sources"][0]["raw_normalized_examples"] >= 1
    assert manifest["sources"][0]["validation_mode"] == "fallback_raw_schema"


def test_run_download_sft_rejects_local_jsonl_sources(monkeypatch, tmp_path: Path):
    cfg = ProjectConfig.model_validate(
        {
            "model": {
                "vocab_size": 1024,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 2,
                "d_ff": 64,
                "context_length": 128,
                "bias": False,
                "dropout": 0.0,
                "rope_theta": 10000.0,
                "gradient_checkpointing": False,
            },
            "sft": {
                "device": "cpu",
                "base_checkpoint": str(tmp_path / "base.pt"),
                "cache_dir": str(tmp_path / "hf_cache"),
                "raw_dir": str(tmp_path / "raw"),
                "prepared_dir": str(tmp_path / "prepared"),
                "sources": [
                    {
                        "name": "custom",
                        "loader": "local_jsonl",
                        "path": str(tmp_path / "custom.jsonl"),
                        "target_examples": 1,
                    }
                ],
                "decontam_datasets": [],
            },
        }
    )

    monkeypatch.setattr("src.posttraining.download.build_tokenizer", lambda: DummyTokenizer())
    monkeypatch.setattr("src.posttraining.download.download_fasttext_langdetect", lambda: None)

    with pytest.raises(ValueError, match="local_jsonl"):
        run_download_sft(cfg)
