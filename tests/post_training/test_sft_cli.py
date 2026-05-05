"""Tests for the SFT CLI checkpoint resolution.

The SFT stage needs a base pretraining checkpoint. There are two sources:
the `--checkpoint` CLI flag and the `sft.base_checkpoint` YAML field.
The CLI flag takes precedence so a user can override a YAML default for
ad-hoc experiments without editing config.
"""

from __future__ import annotations

import pytest

from main import _resolve_sft_checkpoint


def test_cli_takes_precedence_over_yaml():
    """CLI --checkpoint always wins. Lets users override a YAML default."""
    assert _resolve_sft_checkpoint("cli.pt", "yaml.pt") == "cli.pt"


def test_yaml_used_when_cli_omitted():
    """No CLI flag → use the YAML default. Common for the standard run."""
    assert _resolve_sft_checkpoint(None, "yaml.pt") == "yaml.pt"


def test_raises_when_both_missing():
    """Neither source provided → fail loud. SFT on random weights is
    meaningless (VL07 slide 12)."""
    with pytest.raises(ValueError, match="SFT requires"):
        _resolve_sft_checkpoint(None, None)


def test_empty_string_treated_as_unset():
    """An empty string from argparse should fall through to the YAML."""
    assert _resolve_sft_checkpoint("", "yaml.pt") == "yaml.pt"
