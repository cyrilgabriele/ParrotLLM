"""Persistent registry of leaderboard benchmark results.

Each result is keyed by (git_sha, checkpoint_basename, tier) and lives at:
    runs/benchmarks/<git_sha>__<checkpoint_basename>__<tier>.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class BenchmarkResult:
    git_sha: str
    checkpoint_basename: str
    tier: str  # one of: "smoke", "quick", "full"
    scores: dict[str, float]  # per-benchmark accuracy (e.g. {"hellaswag": 33.5, ...})
    pii_named: float           # sum of the four named benchmark scores
    wall_clock_seconds: float


def _filename_for(result: BenchmarkResult) -> str:
    return f"{result.git_sha}__{result.checkpoint_basename}__{result.tier}.json"


def save_result(result: BenchmarkResult, registry_dir: Path) -> Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / _filename_for(result)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
    return path


def load_results(registry_dir: Path) -> list[BenchmarkResult]:
    if not registry_dir.exists():
        return []
    out: list[BenchmarkResult] = []
    for path in sorted(registry_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        out.append(BenchmarkResult(**payload))
    return out
