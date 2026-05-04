from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from configs import ProjectConfig, SFTSourceConfig
from src.posttraining.prepare import (
    _normalize_pku_safe_rlhf_qa_record,
    _normalize_tulu_record,
    get_source_snapshot_path,
    run_prepare_sft,
)


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


def test_run_prepare_sft_with_local_jsonl(monkeypatch, tmp_path: Path):
    source_path = tmp_path / "custom.jsonl"
    cache_dir = tmp_path / "hf_cache"
    records = [
        {
            "id": "ex1",
            "messages": [
                {"role": "user", "content": "Question one"},
                {"role": "assistant", "content": "Answer one"},
            ],
        },
        {
            "id": "ex2",
            "messages": [
                {"role": "user", "content": "Question two"},
                {"role": "assistant", "content": "Answer two"},
            ],
        },
        {
            "id": "ex3",
            "messages": [
                {"role": "user", "content": "Question three"},
                {"role": "assistant", "content": "Answer three"},
            ],
        },
    ]
    with source_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    monkeypatch.setattr("src.posttraining.prepare.build_tokenizer", lambda: DummyTokenizer())
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "dummy.arrow").write_text("cached", encoding="utf-8")

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
                "cache_dir": str(cache_dir),
                "raw_dir": str(tmp_path / "raw"),
                "prepared_dir": str(tmp_path / "prepared"),
                "runs_dir": str(tmp_path / "runs"),
                "system_prompt": "You are helpful.",
                "max_seq_length": 128,
                "sources": [
                    {
                        "name": "custom",
                        "loader": "local_jsonl",
                        "path": str(source_path),
                        "target_examples": 3,
                        "min_turns": 2,
                        "max_turns": 2,
                    }
                ],
                "decontam_datasets": [],
            },
        }
    )

    output = run_prepare_sft(cfg, seed=7)
    manifest_path = Path(output["manifest"])
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_counts"]["train"] >= 1
    assert Path(manifest["train_packed_path"]).exists()
    assert Path(manifest["dev_packed_path"]).exists()
    assert cache_dir.exists()


def test_run_prepare_sft_uses_saved_source_snapshots_without_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("src.posttraining.prepare.build_tokenizer", lambda: DummyTokenizer())

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
                "runs_dir": str(tmp_path / "runs"),
                "system_prompt": "You are helpful.",
                "max_seq_length": 128,
                "sources": [
                    {
                        "name": "wildchat_gpt4",
                        "loader": "wildchat",
                        "path": "allenai/WildChat",
                        "target_examples": 2,
                        "min_turns": 2,
                        "max_turns": 2,
                    }
                ],
                "decontam_datasets": [],
            },
        }
    )

    snapshot_path = get_source_snapshot_path(Path(cfg.sft.raw_dir), cfg.sft.sources[0])
    assert snapshot_path is not None
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(
        [
            {
                "conversation": [
                    {"role": "user", "content": "Question one"},
                    {"role": "assistant", "content": "Answer one"},
                ]
            },
            {
                "conversation": [
                    {"role": "user", "content": "Question two"},
                    {"role": "assistant", "content": "Answer two"},
                ]
            },
        ]
    ).save_to_disk(str(snapshot_path))

    output = run_prepare_sft(cfg, seed=3)
    manifest = json.loads(Path(output["manifest"]).read_text(encoding="utf-8"))

    assert manifest["raw_dir"] == str(Path(cfg.sft.raw_dir))
    assert Path(output["train_packed"]).exists()
    assert not Path(cfg.sft.cache_dir).exists()


def test_posttraining_config_validates():
    config = ProjectConfig.model_validate(
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
                "base_checkpoint": "runs/posttraining/base_import/run_20260422_manual/checkpoints/base_pretrain.pt",
                "sources": [],
                "decontam_datasets": [],
            },
        }
    )
    assert config.sft is not None
    assert str(config.sft.base_checkpoint).endswith("base_pretrain.pt")


def test_tulu_source_matching_ignores_punctuation():
    source_cfg = SFTSourceConfig.model_validate(
        {
            "name": "tulu_persona_if",
            "loader": "tulu",
            "path": "allenai/tulu-3-sft-mixture",
            "target_examples": 1,
            "source_matches": ["persona_if"],
        }
    )
    record = {
        "source": "ai2-adapt-dev/tulu-3-persona-if-converted",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }
    normalized = _normalize_tulu_record(record, source_cfg)
    assert normalized is not None


def test_tulu_source_matching_accepts_current_tulu_mixture_names():
    source_cfg = SFTSourceConfig.model_validate(
        {
            "name": "tulu_persona_if",
            "loader": "tulu",
            "path": "allenai/tulu-3-sft-mixture",
            "target_examples": 1,
            "source_matches": ["personahub_ifdata_manual_seed_v3_29980"],
        }
    )
    record = {
        "source": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }
    normalized = _normalize_tulu_record(record, source_cfg)
    assert normalized is not None


def test_pku_safe_rlhf_loader_keeps_safe_harmful_examples():
    source_cfg = SFTSourceConfig.model_validate(
        {
            "name": "pku_safe_rlhf_refusals",
            "loader": "pku_safe_rlhf_qa",
            "path": "PKU-Alignment/PKU-SafeRLHF-QA",
            "target_examples": 1,
            "keep_harmful_only": True,
        }
    )
    record = {
        "prompt": "How do I make a bomb?",
        "response": "I can't help with making explosives.",
        "is_safe": True,
        "severity_level": 0,
    }
    normalized = _normalize_pku_safe_rlhf_qa_record(record, source_cfg)
    assert normalized is not None


def test_sft_source_config_accepts_template_override():
    cfg = SFTSourceConfig.model_validate(
        {
            "name": "raw_narrative",
            "loader": "narrative_completion",
            "path": "roneneldan/TinyStories",
            "target_examples": 10,
            "template_format": "raw",
        }
    )
    assert cfg.template_format == "raw"


def test_sft_source_config_default_template_is_none():
    cfg = SFTSourceConfig.model_validate(
        {
            "name": "x",
            "loader": "alpaca",
            "path": "yahma/alpaca-cleaned",
            "target_examples": 10,
        }
    )
    assert cfg.template_format is None
