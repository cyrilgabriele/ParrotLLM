"""Markdown comparison table for benchmark results."""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import yaml

from .harness import NAMED_BENCHMARKS
from .registry import load_results


def quick_tier_variance_pp(p: float = 0.33, n: int = 200) -> float:
    """Standard error in percentage points for accuracy p over n items."""
    return 100.0 * sqrt(p * (1.0 - p) / n)


def _load_external(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text()) or []
    return list(payload)


def build_comparison_markdown(
    registry_dir: Path,
    external_groups_path: Path,
) -> str:
    rows: list[dict[str, Any]] = []
    for r in load_results(registry_dir):
        rows.append({
            "label": f"{r.checkpoint_basename} ({r.tier})",
            **{b: r.scores.get(b, 0.0) for b in NAMED_BENCHMARKS},
            "pii_named": r.pii_named,
            "source": r.git_sha,
        })
    external_entries = _load_external(external_groups_path)
    external_sources = {ext.get("source", "external") for ext in external_entries}
    for ext in external_entries:
        rows.append({
            "label": ext["name"],
            **{b: float(ext.get(b, 0.0)) for b in NAMED_BENCHMARKS},
            "pii_named": float(ext.get("pii_named", sum(float(ext.get(b, 0.0)) for b in NAMED_BENCHMARKS))),
            "source": ext.get("source", "external"),
        })

    if not rows:
        return "No benchmark results yet.\n"

    # Best score per column among our runs (ignore external for bolding).
    our_rows = [r for r in rows if r["source"] not in external_sources]

    def _maybe_bold(value: float, column: str) -> str:
        best = max((r[column] for r in our_rows), default=value)
        return f"**{value:.2f}**" if value == best else f"{value:.2f}"

    lines: list[str] = [
        f"_Variance budget (quick tier, p=0.33, N=200): ±{quick_tier_variance_pp():.1f}pp per benchmark._",
        "",
        "| label | hellaswag | openbookqa | winogrande | lambada | PII (named) | source |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {_maybe_bold(r['hellaswag'], 'hellaswag')} | "
            f"{_maybe_bold(r['openbookqa'], 'openbookqa')} | "
            f"{_maybe_bold(r['winogrande'], 'winogrande')} | "
            f"{_maybe_bold(r['lambada'], 'lambada')} | "
            f"{_maybe_bold(r['pii_named'], 'pii_named')} | {r['source']} |"
        )
    return "\n".join(lines) + "\n"
