# Training Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-frontend training dashboard (Gradio, Jupyter widget, Rich TUI) that works in every GPU cluster access scenario and shows per-GPU stats across all connected graphics cards.

**Architecture:** A `src/dashboard/` package with five backend modules (`metrics_reader`, `system_monitor`, `problem_detector`, `run_manager`, `plots`) shared by three frontends (`app.py` Gradio, `notebook.py` Jupyter widget, `tui.py` Rich TUI). All modes launched from `main.py --stage dashboard` with optional `--open`, `--share`, `--tui`, `--tui-refresh` flags.

**Tech Stack:** Gradio 5 (installed), matplotlib (installed), ipywidgets>=8.0 (new), rich>=13.0 (new), psutil (new), pynvml (new).

**Note:** This plan supersedes `local_files/2026-04-05-training-dashboard.md`. Do not implement that plan.

**Spec:** `docs/superpowers/specs/2026-04-05-training-dashboard-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add psutil, pynvml, ipywidgets, rich |
| `src/dashboard/__init__.py` | Create | Package marker |
| `src/dashboard/metrics_reader.py` | Create | Parse metrics.jsonl → TrainingMetrics + stale detection |
| `src/dashboard/system_monitor.py` | Create | CPU/RAM/per-GPU stats via psutil + pynvml |
| `src/dashboard/problem_detector.py` | Create | Analyze TrainingMetrics → list[Alert] |
| `src/dashboard/run_manager.py` | Create | List runs, launch/resume/kill training, tail log |
| `src/dashboard/plots.py` | Create | matplotlib figures, light theme |
| `src/dashboard/app.py` | Create | Gradio UI: Live Monitor, Architecture, Run Manager |
| `src/dashboard/notebook.py` | Create | Jupyter ipywidgets UI (Phase 1 read-only, Phase 2 run mgmt) |
| `src/dashboard/tui.py` | Create | Rich TUI: read-only terminal dashboard |
| `main.py` | Modify | Add dashboard stage + --open, --share, --tui, --tui-refresh flags |
| `tests/dashboard/__init__.py` | Create | Test package marker |
| `tests/dashboard/test_metrics_reader.py` | Create | MetricsReader tests |
| `tests/dashboard/test_system_monitor.py` | Create | SystemMonitor tests |
| `tests/dashboard/test_problem_detector.py` | Create | ProblemDetector tests |
| `tests/dashboard/test_run_manager.py` | Create | RunManager tests |

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add four new dependencies**

In `pyproject.toml`, add these four lines to the `dependencies` list after `"python-dotenv>=1.2.2",`:

```toml
    "psutil>=6.0.0",
    "pynvml>=11.0.0",
    "ipywidgets>=8.0",
    "rich>=13.0",
```

The full dependencies block should now end with:
```toml
    "python-dotenv>=1.2.2",
    "psutil>=6.0.0",
    "pynvml>=11.0.0",
    "ipywidgets>=8.0",
    "rich>=13.0",
]
```

- [ ] **Step 2: Install**

```bash
uv sync
```

Expected: all four packages install without error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(dashboard): add psutil, pynvml, ipywidgets, rich dependencies"
```

---

## Task 2: Package structure

**Files:**
- Create: `src/dashboard/__init__.py`
- Create: `tests/dashboard/__init__.py`

- [ ] **Step 1: Create package markers**

Create `src/dashboard/__init__.py` as an empty file.
Create `tests/dashboard/__init__.py` as an empty file.

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "import src.dashboard; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/__init__.py tests/dashboard/__init__.py
git commit -m "feat(dashboard): create src/dashboard package"
```

---

## Task 3: MetricsReader

**Files:**
- Create: `src/dashboard/metrics_reader.py`
- Create: `tests/dashboard/test_metrics_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_metrics_reader.py
import json
import time
import pytest
from pathlib import Path
from src.dashboard.metrics_reader import read_metrics, TrainingMetrics, is_metrics_stale


@pytest.fixture
def run_dir(tmp_path):
    lines = [
        {"stage": "train", "type": "model_architecture", "vocab_size": 50257,
         "n_layers": 16, "n_heads": 8, "d_model": 320, "d_ff": 854,
         "total_params": 35763840, "trainable_params": 35763840},
        {"stage": "train", "type": "config", "max_steps": 10000,
         "batch_size": 64, "context_length": 1024, "gradient_accumulation_steps": 4},
        {"stage": "train", "type": "step", "step": 100, "train_loss": 4.5,
         "lr": 3e-4, "perplexity": 90.0, "grad_norm": 0.8, "tokens_per_sec": 12000},
        {"stage": "train", "type": "step", "step": 200, "train_loss": 4.1,
         "lr": 3e-4, "perplexity": 60.0, "grad_norm": 0.7, "tokens_per_sec": 12000},
        {"stage": "train", "type": "eval", "step": 200, "val_loss": 4.3,
         "val_ppl": 73.7, "eval_train_loss": 4.1, "eval_train_ppl": 60.3},
        {"stage": "train", "type": "best_checkpoint", "step": 200},
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    return tmp_path


def test_read_metrics_step_data(run_dir):
    m = read_metrics(run_dir)
    assert m.steps == [100, 200]
    assert m.train_losses == pytest.approx([4.5, 4.1])
    assert m.grad_norms == pytest.approx([0.8, 0.7])
    assert m.lrs == pytest.approx([3e-4, 3e-4])
    assert m.tokens_per_sec == pytest.approx([12000, 12000])


def test_read_metrics_eval_data(run_dir):
    m = read_metrics(run_dir)
    assert m.eval_steps == [200]
    assert m.val_losses == pytest.approx([4.3])
    assert m.val_ppls == pytest.approx([73.7])
    assert m.eval_train_losses == pytest.approx([4.1])


def test_read_metrics_architecture(run_dir):
    m = read_metrics(run_dir)
    assert m.architecture["n_layers"] == 16
    assert m.architecture["total_params"] == 35763840


def test_read_metrics_config(run_dir):
    m = read_metrics(run_dir)
    assert m.config["max_steps"] == 10000
    assert m.config["batch_size"] == 64


def test_read_metrics_best_step(run_dir):
    m = read_metrics(run_dir)
    assert m.best_step == 200


def test_read_metrics_missing_file(tmp_path):
    m = read_metrics(tmp_path)
    assert m.steps == []
    assert m.architecture == {}


def test_read_metrics_partial_fields(tmp_path):
    line = {"stage": "train", "type": "step", "step": 50, "train_loss": 5.0,
            "lr": 1e-4, "perplexity": 148.0, "grad_norm": 1.2}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(line))
    m = read_metrics(tmp_path)
    assert m.steps == [50]
    assert m.tokens_per_sec == []


def test_is_metrics_stale_fresh(run_dir):
    stale, age = is_metrics_stale(run_dir, threshold=60)
    assert stale is False
    assert age < 5


def test_is_metrics_stale_missing(tmp_path):
    stale, age = is_metrics_stale(tmp_path)
    assert stale is False
    assert age == 0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_metrics_reader.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.dashboard.metrics_reader'`

- [ ] **Step 3: Implement metrics_reader.py**

```python
# src/dashboard/metrics_reader.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingMetrics:
    steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    tokens_per_sec: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_ppls: list[float] = field(default_factory=list)
    eval_train_losses: list[float] = field(default_factory=list)
    architecture: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    best_step: Optional[int] = None


def read_metrics(run_dir: Path) -> TrainingMetrics:
    """Parse metrics.jsonl from a run directory. Returns empty TrainingMetrics if missing."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return TrainingMetrics()

    m = TrainingMetrics()
    for raw in metrics_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        t = entry.get("type", "")
        if t == "model_architecture":
            m.architecture = {k: v for k, v in entry.items()
                              if k not in ("stage", "type", "timestamp")}
        elif t == "config":
            m.config = {k: v for k, v in entry.items()
                        if k not in ("stage", "type", "timestamp")}
        elif t == "step":
            m.steps.append(entry["step"])
            m.train_losses.append(entry["train_loss"])
            m.lrs.append(entry["lr"])
            m.grad_norms.append(entry["grad_norm"])
            if "tokens_per_sec" in entry:
                m.tokens_per_sec.append(entry["tokens_per_sec"])
        elif t in ("eval", "initial_validation"):
            if "step" in entry:
                m.eval_steps.append(entry["step"])
            if "val_loss" in entry:
                m.val_losses.append(entry["val_loss"])
            if "val_ppl" in entry:
                m.val_ppls.append(entry["val_ppl"])
            if "eval_train_loss" in entry:
                m.eval_train_losses.append(entry["eval_train_loss"])
        elif t == "best_checkpoint":
            m.best_step = entry.get("step")

    return m


def is_metrics_stale(run_dir: Path, threshold: int = 60) -> tuple[bool, int]:
    """Return (is_stale, seconds_since_update). Returns (False, 0) if file absent."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return False, 0
    age = int(time.time() - metrics_path.stat().st_mtime)
    return age > threshold, age
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_metrics_reader.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/metrics_reader.py tests/dashboard/test_metrics_reader.py
git commit -m "feat(dashboard): add MetricsReader — parses metrics.jsonl + stale detection"
```

---

## Task 4: SystemMonitor (multi-GPU)

**Files:**
- Create: `src/dashboard/system_monitor.py`
- Create: `tests/dashboard/test_system_monitor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_system_monitor.py
import math
from unittest.mock import MagicMock, patch
import pytest
from src.dashboard.system_monitor import get_system_stats, SystemStats, GPUStats


@patch("src.dashboard.system_monitor.psutil.cpu_percent", return_value=42.5)
@patch("src.dashboard.system_monitor.psutil.virtual_memory")
@patch("src.dashboard.system_monitor._TORCH_AVAILABLE", False)
def test_no_gpu(mock_vmem, mock_cpu):
    mock_vmem.return_value = MagicMock(used=4 * 1024**3, total=16 * 1024**3)
    stats = get_system_stats()
    assert stats.cpu_percent == pytest.approx(42.5)
    assert stats.ram_used_gb == pytest.approx(4.0)
    assert stats.ram_total_gb == pytest.approx(16.0)
    assert stats.gpus == []
    assert stats.gpu_available is False


@patch("src.dashboard.system_monitor.psutil.cpu_percent", return_value=10.0)
@patch("src.dashboard.system_monitor.psutil.virtual_memory")
@patch("src.dashboard.system_monitor._NVML_AVAILABLE", False)
@patch("src.dashboard.system_monitor._TORCH_AVAILABLE", True)
@patch("src.dashboard.system_monitor.torch")
def test_two_gpus_no_nvml(mock_torch, mock_vmem, mock_cpu):
    mock_vmem.return_value = MagicMock(used=2 * 1024**3, total=8 * 1024**3)
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.cuda.memory_allocated.side_effect = [3 * 1024**3, 2 * 1024**3]
    mock_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=8 * 1024**3)
    mock_torch.cuda.get_device_name.return_value = "A100"
    stats = get_system_stats()
    assert len(stats.gpus) == 2
    assert stats.gpus[0].mem_used_gb == pytest.approx(3.0)
    assert stats.gpus[1].mem_used_gb == pytest.approx(2.0)
    assert math.isnan(stats.gpus[0].utilization_pct)
    assert math.isnan(stats.gpus[0].temperature_c)
    assert stats.gpu_total_used_gb == pytest.approx(5.0)
    assert stats.gpu_total_mem_gb == pytest.approx(16.0)
    assert stats.gpu_available is True
    assert math.isnan(stats.gpu_avg_utilization)


@patch("src.dashboard.system_monitor.psutil.cpu_percent", return_value=10.0)
@patch("src.dashboard.system_monitor.psutil.virtual_memory")
@patch("src.dashboard.system_monitor._NVML_AVAILABLE", True)
@patch("src.dashboard.system_monitor._TORCH_AVAILABLE", True)
@patch("src.dashboard.system_monitor.pynvml")
@patch("src.dashboard.system_monitor.torch")
def test_two_gpus_with_nvml(mock_torch, mock_pynvml, mock_vmem, mock_cpu):
    mock_vmem.return_value = MagicMock(used=2 * 1024**3, total=8 * 1024**3)
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.cuda.memory_allocated.side_effect = [3 * 1024**3, 2 * 1024**3]
    mock_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=8 * 1024**3)
    mock_torch.cuda.get_device_name.return_value = "A100"
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=87)
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 72
    stats = get_system_stats()
    assert stats.gpus[0].utilization_pct == pytest.approx(87.0)
    assert stats.gpus[0].temperature_c == pytest.approx(72.0)
    assert stats.gpu_avg_utilization == pytest.approx(87.0)


@patch("src.dashboard.system_monitor.psutil.cpu_percent", return_value=10.0)
@patch("src.dashboard.system_monitor.psutil.virtual_memory")
@patch("src.dashboard.system_monitor._TORCH_AVAILABLE", True)
@patch("src.dashboard.system_monitor.torch")
def test_cuda_not_available(mock_torch, mock_vmem, mock_cpu):
    mock_vmem.return_value = MagicMock(used=2 * 1024**3, total=8 * 1024**3)
    mock_torch.cuda.is_available.return_value = False
    stats = get_system_stats()
    assert stats.gpus == []
    assert stats.gpu_available is False
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_system_monitor.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.dashboard.system_monitor'`

- [ ] **Step 3: Implement system_monitor.py**

```python
# src/dashboard/system_monitor.py
from __future__ import annotations

import math
from dataclasses import dataclass, field

import psutil

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


@dataclass
class GPUStats:
    index: int
    name: str
    mem_used_gb: float
    mem_total_gb: float
    utilization_pct: float   # float("nan") if pynvml unavailable
    temperature_c: float     # float("nan") if pynvml unavailable


@dataclass
class SystemStats:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpus: list[GPUStats] = field(default_factory=list)
    gpu_total_used_gb: float = 0.0
    gpu_total_mem_gb: float = 0.0
    gpu_avg_utilization: float = float("nan")
    gpu_available: bool = False


def get_system_stats() -> SystemStats:
    """Return current CPU, RAM, and per-GPU stats. Never raises."""
    cpu = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    ram_used = vm.used / 1024**3
    ram_total = vm.total / 1024**3

    gpus: list[GPUStats] = []

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = props.total_memory / 1024**3
            name = torch.cuda.get_device_name(i)

            util = math.nan
            temp = math.nan
            if _NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    temp = float(pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ))
                except Exception:
                    pass

            gpus.append(GPUStats(
                index=i, name=name,
                mem_used_gb=mem_used, mem_total_gb=mem_total,
                utilization_pct=util, temperature_c=temp,
            ))

    total_used = sum(g.mem_used_gb for g in gpus)
    total_mem = sum(g.mem_total_gb for g in gpus)
    utils = [g.utilization_pct for g in gpus if not math.isnan(g.utilization_pct)]
    avg_util = sum(utils) / len(utils) if utils else math.nan

    return SystemStats(
        cpu_percent=cpu,
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        gpus=gpus,
        gpu_total_used_gb=total_used,
        gpu_total_mem_gb=total_mem,
        gpu_avg_utilization=avg_util,
        gpu_available=bool(gpus),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_system_monitor.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/system_monitor.py tests/dashboard/test_system_monitor.py
git commit -m "feat(dashboard): add SystemMonitor — per-GPU stats via pynvml + psutil"
```

---

## Task 5: ProblemDetector

**Files:**
- Create: `src/dashboard/problem_detector.py`
- Create: `tests/dashboard/test_problem_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_problem_detector.py
import pytest
from src.dashboard.metrics_reader import TrainingMetrics
from src.dashboard.problem_detector import detect_problems, Severity


def _make(steps, losses, grad_norms=None, val_losses=None, eval_train_losses=None, lrs=None):
    m = TrainingMetrics()
    m.steps = steps
    m.train_losses = losses
    m.grad_norms = grad_norms or [0.5] * len(steps)
    m.lrs = lrs or [3e-4] * len(steps)
    if val_losses:
        m.eval_steps = steps[-len(val_losses):]
        m.val_losses = val_losses
    if eval_train_losses:
        m.eval_train_losses = eval_train_losses
    return m


def test_no_alerts_clean_run():
    m = _make([100, 200, 300], [4.5, 4.1, 3.8])
    assert detect_problems(m) == []


def test_grad_explosion_detected():
    m = _make([100, 200, 300], [4.5, 4.1, 3.8], grad_norms=[0.5, 15.0, 12.0])
    alerts = detect_problems(m)
    codes = [a.code for a in alerts]
    assert "GRAD_EXPLOSION" in codes
    assert any(a.severity == Severity.ERROR for a in alerts if a.code == "GRAD_EXPLOSION")


def test_no_explosion_below_threshold():
    m = _make([100, 200], [4.5, 4.1], grad_norms=[9.9, 9.8])
    assert all(a.code != "GRAD_EXPLOSION" for a in detect_problems(m))


def test_overfitting_detected():
    m = _make(
        list(range(100, 600, 100)),
        [4.5, 4.3, 4.1, 3.9, 3.8],
        val_losses=[4.6, 4.7, 4.9, 5.2, 5.6],
        eval_train_losses=[4.5, 4.3, 4.1, 3.9, 3.8],
    )
    codes = [a.code for a in detect_problems(m)]
    assert "OVERFITTING" in codes


def test_stagnation_detected():
    m = _make(
        list(range(100, 600, 100)),
        [4.5, 4.3, 4.1, 3.9, 3.8],
        val_losses=[3.5001, 3.5002, 3.5000, 3.5001, 3.5000],
        eval_train_losses=[3.4, 3.4, 3.4, 3.4, 3.4],
    )
    codes = [a.code for a in detect_problems(m)]
    assert "STAGNATION" in codes


def test_high_loss_detected():
    m = _make(list(range(200, 500, 100)), [8.0, 8.1, 7.9])
    codes = [a.code for a in detect_problems(m)]
    assert "HIGH_LOSS" in codes


def test_lr_zero_detected():
    m = _make([100, 200, 300], [4.5, 4.1, 3.8], lrs=[0.0, 0.0, 0.0])
    codes = [a.code for a in detect_problems(m)]
    assert "LR_ZERO" in codes


def test_lr_zero_not_flagged_early():
    m = _make([5, 10], [9.0, 8.5], lrs=[0.0, 0.0])
    assert all(a.code != "LR_ZERO" for a in detect_problems(m))
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_problem_detector.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.dashboard.problem_detector'`

- [ ] **Step 3: Implement problem_detector.py**

```python
# src/dashboard/problem_detector.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.dashboard.metrics_reader import TrainingMetrics


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Alert:
    severity: Severity
    code: str
    message: str
    detail: str


def detect_problems(metrics: TrainingMetrics) -> list[Alert]:
    """Analyze training metrics and return a list of Alerts."""
    alerts: list[Alert] = []

    # GRAD_EXPLOSION: grad_norm > 10.0 in any of the last 3 steps
    if metrics.grad_norms:
        recent = metrics.grad_norms[-3:]
        if any(g > 10.0 for g in recent):
            worst = max(recent)
            alerts.append(Alert(
                severity=Severity.ERROR, code="GRAD_EXPLOSION",
                message="Gradient explosion detected",
                detail=f"Grad norm {worst:.2f} in the last 3 steps (threshold: 10.0). "
                       "Consider reducing lr or increasing grad_clip.",
            ))

    # OVERFITTING: val-train gap > 0.5 AND widening over last 3 evals
    if len(metrics.val_losses) >= 3 and len(metrics.eval_train_losses) >= 3:
        gaps = [v - t for v, t in zip(metrics.val_losses[-3:], metrics.eval_train_losses[-3:])]
        if gaps[-1] > 0.5 and gaps[-1] > gaps[0]:
            alerts.append(Alert(
                severity=Severity.WARNING, code="OVERFITTING",
                message="Overfitting detected",
                detail=f"Val–train gap is {gaps[-1]:.3f} and widening "
                       f"(was {gaps[0]:.3f} three evals ago). "
                       "Consider adding dropout or early stopping.",
            ))

    # STAGNATION: val_loss not improved by > 0.001 in last 5 evals
    if len(metrics.val_losses) >= 5:
        window = metrics.val_losses[-5:]
        if max(window) - min(window) < 0.001:
            alerts.append(Alert(
                severity=Severity.WARNING, code="STAGNATION",
                message="Training stagnation detected",
                detail=f"Val loss range over last 5 evals is only "
                       f"{max(window) - min(window):.5f}. Training may have plateaued.",
            ))

    # HIGH_LOSS: train_loss > 7.0 after step 200
    if metrics.steps and metrics.steps[-1] >= 200:
        recent = [l for s, l in zip(metrics.steps, metrics.train_losses) if s >= 200]
        if recent and all(l > 7.0 for l in recent[-3:]):
            alerts.append(Alert(
                severity=Severity.ERROR, code="HIGH_LOSS",
                message="Abnormally high training loss",
                detail=f"Train loss is {recent[-1]:.2f} after step 200. "
                       "Model may not be learning — check data or lr.",
            ))

    # LR_ZERO: learning rate is 0 after step 20
    if metrics.steps and metrics.steps[-1] > 20:
        if all(lr == 0.0 for lr in metrics.lrs[-3:]):
            alerts.append(Alert(
                severity=Severity.ERROR, code="LR_ZERO",
                message="Learning rate is zero",
                detail="LR has been 0 for the last 3 logged steps after step 20. "
                       "Check scheduler configuration.",
            ))

    return alerts
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_problem_detector.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/problem_detector.py tests/dashboard/test_problem_detector.py
git commit -m "feat(dashboard): add ProblemDetector — GRAD_EXPLOSION, OVERFITTING, STAGNATION, HIGH_LOSS, LR_ZERO"
```

---

## Task 6: RunManager

**Files:**
- Create: `src/dashboard/run_manager.py`
- Create: `tests/dashboard/test_run_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_run_manager.py
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.dashboard.run_manager import (
    list_runs, RunInfo, launch_training, get_latest_run_dir,
    kill_training, get_log_lines,
)


@pytest.fixture
def runs_dir(tmp_path):
    for name, steps, best_val in [
        ("run_20260401_100000", [100, 200], 4.3),
        ("run_20260402_120000", [100, 200, 300], 3.9),
    ]:
        d = tmp_path / name
        d.mkdir()
        lines = [
            {"type": "step", "step": s, "train_loss": 4.0, "lr": 3e-4,
             "grad_norm": 0.5, "perplexity": 55.0}
            for s in steps
        ]
        lines.append({"type": "eval", "step": steps[-1], "val_loss": best_val,
                      "val_ppl": 73.0, "eval_train_loss": 3.9})
        lines.append({"type": "best_checkpoint", "step": steps[-1]})
        (d / "metrics.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    return tmp_path


def test_list_runs_returns_all(runs_dir):
    assert len(list_runs(runs_dir)) == 2


def test_list_runs_sorted_newest_first(runs_dir):
    runs = list_runs(runs_dir)
    assert runs[0].name == "run_20260402_120000"
    assert runs[1].name == "run_20260401_100000"


def test_list_runs_run_info_fields(runs_dir):
    newest = list_runs(runs_dir)[0]
    assert isinstance(newest, RunInfo)
    assert newest.last_step == 300
    assert newest.best_val_loss == pytest.approx(3.9)
    assert newest.run_dir == runs_dir / "run_20260402_120000"


def test_list_runs_empty_dir(tmp_path):
    assert list_runs(tmp_path) == []


def test_get_latest_run_dir(runs_dir):
    assert get_latest_run_dir(runs_dir).name == "run_20260402_120000"


def test_get_latest_run_dir_empty(tmp_path):
    assert get_latest_run_dir(tmp_path) is None


@patch("src.dashboard.run_manager.subprocess.Popen")
def test_launch_training_starts_process(mock_popen, tmp_path):
    mock_popen.return_value = MagicMock(pid=1234)
    launch_training(config_path=Path("configs/default.yaml"))
    call_args = mock_popen.call_args[0][0]
    assert "uv" in call_args
    assert "train" in call_args


@patch("src.dashboard.run_manager.subprocess.Popen")
def test_launch_training_with_resume(mock_popen, tmp_path):
    mock_popen.return_value = MagicMock(pid=5678)
    launch_training(
        config_path=Path("configs/default.yaml"),
        resume_run_dir=Path("runs/run_20260402_120000"),
    )
    call_args = mock_popen.call_args[0][0]
    assert "--resume" in call_args


def test_kill_training_terminates():
    proc = MagicMock()
    proc.poll.return_value = None
    kill_training(proc)
    proc.terminate.assert_called_once()


def test_kill_training_already_dead():
    proc = MagicMock()
    proc.poll.return_value = 0
    kill_training(proc)
    proc.terminate.assert_not_called()


def test_get_log_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"line {i}" for i in range(30)))
    lines = get_log_lines(tmp_path, n=5)
    assert lines == ["line 25", "line 26", "line 27", "line 28", "line 29"]


def test_get_log_lines_missing(tmp_path):
    assert get_log_lines(tmp_path) == []
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_run_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.dashboard.run_manager'`

- [ ] **Step 3: Implement run_manager.py**

```python
# src/dashboard/run_manager.py
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.dashboard.metrics_reader import read_metrics


@dataclass
class RunInfo:
    name: str
    run_dir: Path
    last_step: Optional[int]
    best_val_loss: Optional[float]


def list_runs(runs_dir: Path) -> list[RunInfo]:
    """Return all runs sorted newest-first by directory name."""
    if not runs_dir.exists():
        return []
    dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    result = []
    for d in dirs:
        m = read_metrics(d)
        result.append(RunInfo(
            name=d.name,
            run_dir=d,
            last_step=m.steps[-1] if m.steps else None,
            best_val_loss=m.val_losses[m.eval_steps.index(m.best_step)]
                if m.best_step and m.best_step in m.eval_steps and m.val_losses else None,
        ))
    return result


def get_latest_run_dir(runs_dir: Path) -> Optional[Path]:
    """Return the most recent run directory, or None if none exist."""
    runs = list_runs(runs_dir)
    return runs[0].run_dir if runs else None


def launch_training(
    config_path: Path,
    resume_run_dir: Optional[Path] = None,
) -> subprocess.Popen:
    """Launch training as a subprocess. Returns the Popen handle."""
    cmd = [
        "uv", "run", "python", "main.py",
        "--stage", "train",
        "--config", str(config_path),
    ]
    if resume_run_dir is not None:
        cmd += ["--resume", str(resume_run_dir)]
    return subprocess.Popen(cmd)


def kill_training(proc: subprocess.Popen) -> None:
    """Send SIGTERM to proc if it is still alive."""
    if proc is not None and proc.poll() is None:
        proc.terminate()


def get_log_lines(run_dir: Path, n: int = 20) -> list[str]:
    """Return the last n lines from train.log in run_dir, or [] if unavailable."""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_run_manager.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Run full dashboard test suite so far**

```bash
uv run pytest tests/dashboard/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/run_manager.py tests/dashboard/test_run_manager.py
git commit -m "feat(dashboard): add RunManager — list/launch/resume/kill training, tail log"
```

---

## Task 7: plots.py (light theme)

**Files:**
- Create: `src/dashboard/plots.py`

No unit test — matplotlib figure output is fragile to test and the function is thin glue
over the existing `build_figure` logic. Validated visually in Task 8 smoke test.

- [ ] **Step 1: Create plots.py**

```python
# src/dashboard/plots.py
"""Generate matplotlib figures for the dashboard (light theme)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

from src.dashboard.metrics_reader import TrainingMetrics

TRAIN_COLOR = "#2563EB"   # blue
VAL_COLOR   = "#EA580C"   # orange
LR_COLOR    = "#16A34A"   # green
GRAD_COLOR  = "#D97706"   # amber


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def build_training_figure(metrics: TrainingMetrics) -> Optional[plt.Figure]:
    """Return a 2×2 Figure from TrainingMetrics, or None if no data."""
    if not metrics.steps:
        return None

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    steps = metrics.steps
    eval_steps = metrics.eval_steps

    # ── [0,0] Train & Val Loss ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, metrics.train_losses, color=TRAIN_COLOR, linewidth=1.5, label="Train")
    if eval_steps and metrics.val_losses:
        ax1.plot(eval_steps, metrics.val_losses, color=VAL_COLOR, linewidth=1.5,
                 linestyle="--", label="Val")
    if metrics.best_step:
        ax1.axvline(metrics.best_step, color="#999", linewidth=0.8, linestyle=":",
                    label=f"Best @ {metrics.best_step}")
    _style(ax1)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Validation Loss")
    ax1.legend(fontsize=7, frameon=False)

    # ── [0,1] Validation Perplexity ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if eval_steps and metrics.val_ppls:
        ax2.plot(eval_steps, metrics.val_ppls, color=VAL_COLOR, linewidth=1.5)
        ax2.set_yscale("log")
    _style(ax2)
    ax2.set_ylabel("Perplexity (log)")
    ax2.set_title("Validation Perplexity")

    # ── [1,0] Learning Rate + Grad Norm (twin axis) ───────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, metrics.lrs, color=LR_COLOR, linewidth=1.5, label="LR")
    ax3.set_ylabel("Learning Rate", color=LR_COLOR)
    ax3.tick_params(axis="y", labelcolor=LR_COLOR)
    ax3.set_xlabel("Step")
    ax3.set_title("LR & Grad Norm")
    _style(ax3)
    if metrics.grad_norms:
        ax3b = ax3.twinx()
        ax3b.plot(steps, metrics.grad_norms, color=GRAD_COLOR, linewidth=1.0,
                  alpha=0.7, label="Grad Norm")
        ax3b.set_ylabel("Grad Norm", color=GRAD_COLOR)
        ax3b.tick_params(axis="y", labelcolor=GRAD_COLOR)
        ax3b.spines["top"].set_visible(False)

    # ── [1,1] Train–Val Gap ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if eval_steps and metrics.val_losses and metrics.eval_train_losses:
        gap = [v - t for v, t in zip(metrics.val_losses, metrics.eval_train_losses)]
        ax4.plot(eval_steps, gap, color=VAL_COLOR, linewidth=1.5)
        ax4.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
    _style(ax4)
    ax4.set_ylabel("Val − Train Loss")
    ax4.set_xlabel("Step")
    ax4.set_title("Generalization Gap")

    fig.tight_layout(pad=1.5)
    return fig
```

- [ ] **Step 2: Smoke-test that the figure renders without error**

```bash
uv run python -c "
from src.dashboard.metrics_reader import TrainingMetrics
from src.dashboard.plots import build_training_figure
m = TrainingMetrics()
assert build_training_figure(m) is None
m.steps = [1, 2, 3]
m.train_losses = [4.5, 4.1, 3.8]
m.lrs = [1e-4, 1e-4, 1e-4]
m.grad_norms = [0.8, 0.7, 0.6]
fig = build_training_figure(m)
assert fig is not None
print('ok')
"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/plots.py
git commit -m "feat(dashboard): add plots.py — light-theme 2x2 figure"
```

---

## Task 8: Gradio app (app.py)

**Files:**
- Create: `src/dashboard/app.py`

No automated unit test for the Gradio layout — validated by smoke-launch in Step 2.

- [ ] **Step 1: Create app.py**

```python
# src/dashboard/app.py
"""ParrotLLM Training Dashboard — Gradio UI."""
from __future__ import annotations

import io
import math
import threading
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")

from src.dashboard.metrics_reader import read_metrics, TrainingMetrics, is_metrics_stale
from src.dashboard.system_monitor import get_system_stats
from src.dashboard.problem_detector import detect_problems, Severity
from src.dashboard.run_manager import (
    list_runs, launch_training, get_latest_run_dir,
    kill_training, get_log_lines,
)
from src.dashboard.plots import build_training_figure

_active_proc: Optional[object] = None
_proc_lock = threading.Lock()

_SEVERITY_EMOJI = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_choices(runs_dir: Path) -> list[str]:
    return [r.name for r in list_runs(runs_dir)]


def _selected(runs_dir: Path, run_name: str) -> tuple[TrainingMetrics, Optional[Path]]:
    run_dir = (runs_dir / run_name) if run_name else get_latest_run_dir(runs_dir)
    if run_dir is None:
        return TrainingMetrics(), None
    return read_metrics(run_dir), run_dir


def _fmt_progress(metrics: TrainingMetrics, run_dir: Optional[Path]) -> str:
    if not metrics.steps:
        return "No training data yet. Start training with --stage train."
    current = metrics.steps[-1]
    max_steps = metrics.config.get("max_steps")
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    parts = [f"Step {current:,}"]
    if max_steps:
        parts.append(f"/ {max_steps:,} ({100.0 * current / max_steps:.1f}%)")
    parts.append(f"| Loss {loss:.4f} | LR {lr:.2e}")
    if metrics.val_losses:
        parts.append(f"| Val {metrics.val_losses[-1]:.4f}")
    if metrics.best_step:
        parts.append(f"| Best @ {metrics.best_step:,}")
    if run_dir is not None:
        stale, age = is_metrics_stale(run_dir)
        if stale:
            parts.append(f"\n⚠ Metrics not updated for {age}s — training may have stalled.")
    return " ".join(parts)


def _compute_eta(metrics: TrainingMetrics) -> str:
    if not metrics.steps or not metrics.tokens_per_sec:
        return "—"
    max_steps = metrics.config.get("max_steps")
    if not max_steps:
        return "—"
    remaining = max_steps - metrics.steps[-1]
    if remaining <= 0:
        return "Done"
    tokens_per_step = (
        metrics.config.get("batch_size", 64)
        * metrics.config.get("context_length", 1024)
        * metrics.config.get("gradient_accumulation_steps", 4)
    )
    avg_tps = sum(metrics.tokens_per_sec[-10:]) / len(metrics.tokens_per_sec[-10:])
    if avg_tps <= 0:
        return "—"
    eta_sec = remaining * tokens_per_step / avg_tps
    h, m = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
    return f"~{h}h {m:02d}m" if h > 0 else f"~{m}m"


def _alert_rows(metrics: TrainingMetrics) -> list[list[str]]:
    alerts = detect_problems(metrics)
    return [[_SEVERITY_EMOJI[a.severity], a.code, a.message] for a in alerts]


def _gpu_rows(stats) -> list[list[str]]:
    if not stats.gpu_available:
        return [["—", "No GPU detected", "—", "—", "—", "—"]]
    rows = []
    for g in stats.gpus:
        util = f"{g.utilization_pct:.0f}" if not math.isnan(g.utilization_pct) else "—"
        temp = f"{g.temperature_c:.0f}" if not math.isnan(g.temperature_c) else "—"
        rows.append([str(g.index), g.name,
                     f"{g.mem_used_gb:.1f}", f"{g.mem_total_gb:.1f}", util, temp])
    avg_util = (f"{stats.gpu_avg_utilization:.0f}"
                if not math.isnan(stats.gpu_avg_utilization) else "—")
    rows.append(["ALL", "", f"{stats.gpu_total_used_gb:.1f}",
                 f"{stats.gpu_total_mem_gb:.1f}", avg_util, "—"])
    return rows


def _is_alive() -> bool:
    with _proc_lock:
        return _active_proc is not None and _active_proc.poll() is None


def _arch_text(metrics: TrainingMetrics) -> str:
    arch = metrics.architecture
    if not arch:
        return "No architecture data available."
    return "\n".join([
        f"Vocab size:       {arch.get('vocab_size', '?'):,}",
        f"d_model:          {arch.get('d_model', '?')}",
        f"Layers:           {arch.get('n_layers', '?')}",
        f"Heads:            {arch.get('n_heads', '?')}",
        f"FFN dim:          {arch.get('d_ff', '?')}",
        f"Total params:     {arch.get('total_params', 0):,}",
        f"Trainable params: {arch.get('trainable_params', 0):,}",
    ])


# ── App builder ───────────────────────────────────────────────────────────────

def build_app(runs_dir: Path, config_path: Path) -> gr.Blocks:
    global _active_proc

    def refresh_monitor(run_name):
        metrics, run_dir = _selected(runs_dir, run_name)
        stats = get_system_stats()
        alert_data = _alert_rows(metrics)
        log_lines = get_log_lines(run_dir) if run_dir else []
        return (
            _fmt_progress(metrics, run_dir),
            _compute_eta(metrics),
            gr.update(value=alert_data, visible=bool(alert_data)),
            f"CPU {stats.cpu_percent:.1f}%  |  RAM {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB",
            _gpu_rows(stats),
            build_training_figure(metrics),
            "\n".join(log_lines) if log_lines else "(no log yet)",
        )

    def generate_pdf(run_name):
        from src.scripts.plot_training import plot_run_dir
        _, run_dir = _selected(runs_dir, run_name)
        if run_dir is None:
            return "No run selected."
        out = plot_run_dir(run_dir)
        return f"Saved: {out}"

    def refresh_arch(run_name):
        metrics, _ = _selected(runs_dir, run_name)
        return _arch_text(metrics), metrics.architecture or {}

    def action_start(_):
        global _active_proc
        with _proc_lock:
            _active_proc = launch_training(config_path=config_path)
        return (f"Started. PID: {_active_proc.pid}",
                gr.update(interactive=False), gr.update(interactive=True))

    def action_resume(run_name):
        global _active_proc
        if not run_name:
            return "Select a run to resume.", gr.update(), gr.update()
        with _proc_lock:
            _active_proc = launch_training(config_path=config_path,
                                           resume_run_dir=runs_dir / run_name)
        return (f"Resumed {run_name}. PID: {_active_proc.pid}",
                gr.update(interactive=False), gr.update(interactive=True))

    def action_stop(_):
        global _active_proc
        with _proc_lock:
            if _active_proc is not None:
                kill_training(_active_proc)
                _active_proc = None
        return "Stopped.", gr.update(interactive=True), gr.update(interactive=False)

    def refresh_status():
        alive = _is_alive()
        pid = _active_proc.pid if alive and _active_proc else None
        status = f"● Running — PID {pid}" if alive else "○ Idle"
        return status, gr.update(interactive=not alive), gr.update(interactive=alive)

    choices = _run_choices(runs_dir)

    with gr.Blocks(
        title="ParrotLLM Training Dashboard",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=".gradio-container { max-width: 1400px; margin: auto; }",
    ) as demo:

        gr.Markdown("# ParrotLLM Training Dashboard")

        with gr.Tabs():

            # ── TAB 1: Live Monitor ───────────────────────────────────
            with gr.Tab("Live Monitor"):
                with gr.Row():
                    run_selector = gr.Dropdown(label="Run", choices=choices, scale=3)
                    refresh_slider = gr.Slider(1, 30, value=5, step=1,
                                               label="Refresh every (s)", scale=2)

                alerts_table = gr.Dataframe(
                    headers=["", "Code", "Message"], label="Alerts", visible=False,
                )

                with gr.Row():
                    progress_box = gr.Textbox(label="Progress", scale=4)
                    eta_box = gr.Textbox(label="ETA", scale=1)

                plot_out = gr.Plot(label="Training Metrics")

                sys_header_box = gr.Textbox(label="System", interactive=False)
                gpu_table = gr.Dataframe(
                    headers=["GPU", "Name", "Mem Used (GB)", "Mem Total (GB)",
                             "Util (%)", "Temp (°C)"],
                    label="GPU Stats",
                )

                with gr.Row():
                    pdf_btn = gr.Button("⬇ Generate PDF", scale=1)
                    pdf_out = gr.Textbox(label="", scale=3, interactive=False)

                with gr.Accordion("Training log (last 20 lines)", open=False):
                    log_box = gr.Textbox(lines=10, show_label=False, interactive=False)

                timer = gr.Timer(value=5)
                timer.tick(
                    fn=refresh_monitor, inputs=[run_selector],
                    outputs=[progress_box, eta_box, alerts_table,
                             sys_header_box, gpu_table, plot_out, log_box],
                )
                refresh_slider.change(
                    fn=lambda v: gr.Timer(value=v),
                    inputs=[refresh_slider], outputs=[timer],
                )
                pdf_btn.click(fn=generate_pdf, inputs=[run_selector], outputs=[pdf_out])

            # ── TAB 2: Architecture ───────────────────────────────────
            with gr.Tab("Architecture"):
                arch_run_selector = gr.Dropdown(label="Run", choices=choices)
                gr.Button("Load").click(
                    fn=refresh_arch, inputs=[arch_run_selector],
                    outputs=["arch_box", "arch_json"],
                )
                arch_box = gr.Textbox(label="Architecture Summary",
                                      lines=10, elem_id="arch_box")
                with gr.Accordion("Raw JSON", open=False):
                    arch_json = gr.JSON(elem_id="arch_json")

            # ── TAB 3: Run Manager ────────────────────────────────────
            with gr.Tab("Run Manager"):
                status_box = gr.Textbox(label="Status", value="○ Idle", interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Start new run**")
                        start_btn = gr.Button("▶ Start Training", variant="primary")
                        start_out = gr.Textbox(label="Output", lines=2, interactive=False)

                    with gr.Column():
                        gr.Markdown("**Resume existing run**")
                        resume_selector = gr.Dropdown(label="Run to Resume", choices=choices)
                        resume_btn = gr.Button("⏩ Resume Training")
                        resume_out = gr.Textbox(label="Output", lines=2, interactive=False)

                    with gr.Column():
                        gr.Markdown("**Stop training**")
                        stop_btn = gr.Button("⏹ Stop Training", variant="stop",
                                             interactive=False)
                        stop_out = gr.Textbox(label="Output", lines=2, interactive=False)

                start_btn.click(fn=action_start, inputs=[start_btn],
                                outputs=[start_out, start_btn, stop_btn])
                resume_btn.click(fn=action_resume, inputs=[resume_selector],
                                 outputs=[resume_out, resume_btn, stop_btn])
                stop_btn.click(fn=action_stop, inputs=[stop_btn],
                               outputs=[stop_out, start_btn, stop_btn])

                gr.Markdown("### All Runs")
                runs_table = gr.Dataframe(
                    headers=["Run", "Last Step", "Best Val Loss", "Status"],
                    value=[[r.name, str(r.last_step or "—"),
                            f"{r.best_val_loss:.4f}" if r.best_val_loss else "—",
                            "unknown"]
                           for r in list_runs(runs_dir)],
                    label="Runs",
                )

                status_timer = gr.Timer(value=5)
                status_timer.tick(fn=refresh_status, inputs=[],
                                  outputs=[status_box, start_btn, stop_btn])

    return demo


def run_dashboard(
    runs_dir: Path,
    config_path: Path,
    port: int = 7861,
    share: bool = False,
    open_browser: bool = False,
) -> None:
    demo = build_app(runs_dir, config_path)
    if open_browser:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
```

- [ ] **Step 2: Smoke-test that app builds without error**

```bash
uv run python -c "
from pathlib import Path
from src.dashboard.app import build_app
demo = build_app(Path('runs'), Path('configs/default.yaml'))
print('build_app ok')
"
```

Expected: `build_app ok` with no exceptions.

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat(dashboard): add Gradio app — Live Monitor, Architecture, Run Manager"
```

---

## Task 9: Wire dashboard into main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add dashboard to --stage choices**

In `main.py`, change line 18 from:
```python
        choices=["preprocess", "train", "tune", "eval", "inference", "chat"],
```
to:
```python
        choices=["preprocess", "train", "tune", "eval", "inference", "chat", "dashboard"],
```

- [ ] **Step 2: Add --open, --share, --tui, --tui-refresh flags**

After the `--mock-testing` argument (around line 38), add:

```python
    # dashboard-specific
    parser.add_argument("--open", action="store_true",
                        help="Open browser automatically when starting the dashboard")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share URL")
    parser.add_argument("--tui", action="store_true",
                        help="Use terminal UI instead of Gradio")
    parser.add_argument("--tui-refresh", type=int, default=5, metavar="N",
                        help="TUI refresh interval in seconds (default: 5)")
```

- [ ] **Step 3: Add dashboard handler**

After the `if args.stage == "chat":` block (around line 145, before the closing of `main()`), add:

```python
    if args.stage == "dashboard":
        from pathlib import Path as _Path
        training_cfg = project_config.training
        runs_dir = _Path(training_cfg.runs_dir) if training_cfg else _Path("runs")

        if args.tui:
            from src.dashboard.tui import run_tui
            run_tui(runs_dir=runs_dir, refresh=args.tui_refresh)
        else:
            from src.dashboard.app import run_dashboard
            run_dashboard(
                runs_dir=runs_dir,
                config_path=args.config,
                share=args.share,
                open_browser=args.open,
            )
        return
```

- [ ] **Step 4: Verify the CLI accepts the new stage**

```bash
uv run python main.py --stage dashboard --help
```

Expected: help text shows `--open`, `--share`, `--tui`, `--tui-refresh` without errors.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(dashboard): wire dashboard into main.py — --open, --share, --tui flags"
```

---

## Task 10: Jupyter widget Phase 1 (read-only monitor)

**Files:**
- Create: `src/dashboard/notebook.py`

No automated test — ipywidgets requires a live kernel. Validated by running `monitor()` in
a Jupyter notebook connected to a live kernel.

- [ ] **Step 1: Create notebook.py**

```python
# src/dashboard/notebook.py
"""ParrotLLM Training Dashboard — Jupyter ipywidgets UI.

Phase 1: read-only monitoring (auto-refresh, run selector, GPU table, alerts, plot).
Phase 2: run management buttons (see bottom of file).

Usage:
    from src.dashboard.notebook import monitor
    monitor()                          # auto-detects latest run
    monitor(run_dir="runs/20260405_…") # specific run
    monitor(refresh=10)                # custom refresh interval in seconds
"""
from __future__ import annotations

import io
import math
import threading
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, Image, clear_output

from src.dashboard.metrics_reader import read_metrics, TrainingMetrics, is_metrics_stale
from src.dashboard.system_monitor import get_system_stats
from src.dashboard.problem_detector import detect_problems, Severity
from src.dashboard.run_manager import list_runs, get_latest_run_dir
from src.dashboard.plots import build_training_figure

_SEVERITY_EMOJI = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _metrics_html(metrics: TrainingMetrics, run_dir: Optional[Path]) -> str:
    if not metrics.steps:
        return "<i>No training data yet. Start training first.</i>"
    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    parts = [f"<b>Step</b> {step:,}", f"<b>Train Loss</b> {loss:.4f}", f"<b>LR</b> {lr:.2e}"]
    if metrics.val_losses:
        parts.append(f"<b>Val</b> {metrics.val_losses[-1]:.4f}")
    if metrics.grad_norms:
        parts.append(f"<b>Grad Norm</b> {metrics.grad_norms[-1]:.3f}")
    if metrics.tokens_per_sec:
        parts.append(f"<b>Tok/s</b> {metrics.tokens_per_sec[-1]:,.0f}")
    if metrics.best_step:
        parts.append(f"<b>Best Step</b> {metrics.best_step:,}")
    html = "  │  ".join(parts)
    if run_dir is not None:
        stale, age = is_metrics_stale(run_dir)
        if stale:
            html += f"<br><span style='color:orange'>⚠ Metrics not updated for {age}s</span>"
    return html


def _gpu_html(stats) -> str:
    header = (f"<b>CPU</b> {stats.cpu_percent:.1f}%  │  "
              f"<b>RAM</b> {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB")
    if not stats.gpu_available:
        return header + "  │  GPU: N/A"
    rows = []
    for g in stats.gpus:
        util = f"{g.utilization_pct:.0f}%" if not math.isnan(g.utilization_pct) else "—"
        temp = f"{g.temperature_c:.0f}°C" if not math.isnan(g.temperature_c) else "—"
        rows.append(f"<tr><td>{g.index}</td><td>{g.name}</td>"
                    f"<td>{g.mem_used_gb:.1f} GB</td><td>{g.mem_total_gb:.1f} GB</td>"
                    f"<td>{util}</td><td>{temp}</td></tr>")
    avg_util = (f"{stats.gpu_avg_utilization:.0f}%"
                if not math.isnan(stats.gpu_avg_utilization) else "—")
    rows.append(f"<tr><td><b>ALL</b></td><td></td>"
                f"<td><b>{stats.gpu_total_used_gb:.1f} GB</b></td>"
                f"<td><b>{stats.gpu_total_mem_gb:.1f} GB</b></td>"
                f"<td><b>{avg_util}</b></td><td>—</td></tr>")
    table = (
        "<table border='1' style='border-collapse:collapse;font-size:12px;margin-top:4px'>"
        "<tr><th>GPU</th><th>Name</th><th>Mem Used</th><th>Mem Total</th>"
        "<th>Util</th><th>Temp</th></tr>"
        + "".join(rows) + "</table>"
    )
    return header + "<br>" + table


def _alerts_html(metrics: TrainingMetrics) -> str:
    alerts = detect_problems(metrics)
    if not alerts:
        return ""
    lines = [f"{_SEVERITY_EMOJI[a.severity]} <b>{a.code}</b> — {a.message}" for a in alerts]
    return (
        "<div style='background:#fff3cd;padding:6px;border-radius:4px;margin:4px 0'>"
        + "<br>".join(lines) + "</div>"
    )


class _Monitor:
    """Phase 1 read-only monitor widget."""

    def __init__(self, runs_dir: Path, run_dir: Optional[Path], refresh: int):
        self._runs_dir = runs_dir
        self._refresh = refresh
        self._timer: Optional[threading.Timer] = None
        self._stopped = False

        runs = list_runs(runs_dir)
        run_names = [r.name for r in runs]
        initial = run_dir.name if run_dir and run_dir.name in run_names else (
            run_names[0] if run_names else ""
        )

        self._dropdown = widgets.Dropdown(options=run_names, value=initial,
                                          description="Run:", layout=widgets.Layout(width="300px"))
        self._stop_btn = widgets.Button(description="■ Stop refresh",
                                        button_style="warning",
                                        layout=widgets.Layout(width="140px"))
        self._metrics_w = widgets.HTML()
        self._gpu_w = widgets.HTML()
        self._alerts_w = widgets.HTML()
        self._plot_w = widgets.Output()

        self._stop_btn.on_click(lambda _: self._stop())
        self._dropdown.observe(
            lambda change: self._refresh_data() if change["name"] == "value" else None
        )

        self._refresh_data()
        self._schedule()

    def _get_run_dir(self) -> Optional[Path]:
        val = self._dropdown.value
        if val:
            return self._runs_dir / val
        return get_latest_run_dir(self._runs_dir)

    def _refresh_data(self):
        run_dir = self._get_run_dir()
        if run_dir is None:
            self._metrics_w.value = "<i>No runs found. Start training first.</i>"
            return
        metrics = read_metrics(run_dir)
        stats = get_system_stats()
        self._metrics_w.value = _metrics_html(metrics, run_dir)
        self._gpu_w.value = _gpu_html(stats)
        alerts = _alerts_html(metrics)
        self._alerts_w.value = alerts
        self._alerts_w.layout.display = "" if alerts else "none"
        fig = build_training_figure(metrics)
        with self._plot_w:
            clear_output(wait=True)
            if fig:
                display(Image(data=_fig_to_png(fig)))

    def _schedule(self):
        if not self._stopped:
            self._timer = threading.Timer(self._refresh, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self):
        self._refresh_data()
        self._schedule()

    def _stop(self):
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self._stop_btn.description = "■ Stopped"
        self._stop_btn.disabled = True

    def widget(self) -> widgets.VBox:
        header = widgets.HBox([self._dropdown, self._stop_btn])
        return widgets.VBox([
            header,
            self._metrics_w,
            self._gpu_w,
            self._alerts_w,
            self._plot_w,
        ])


def monitor(
    runs_dir: str | Path = "runs",
    run_dir: Optional[str | Path] = None,
    refresh: int = 5,
) -> None:
    """Display the ParrotLLM training monitor widget in a Jupyter notebook.

    Args:
        runs_dir: Directory containing run subdirectories. Default: "runs".
        run_dir:  Specific run directory to show. Default: auto-detects latest.
        refresh:  Auto-refresh interval in seconds. Default: 5.
    """
    runs_dir = Path(runs_dir)
    run_dir = Path(run_dir) if run_dir is not None else None
    m = _Monitor(runs_dir, run_dir, refresh)
    display(m.widget())
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
uv run python -c "from src.dashboard.notebook import monitor; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/notebook.py
git commit -m "feat(dashboard): add Jupyter widget Phase 1 — read-only monitor()"
```

---

## Task 11: Jupyter widget Phase 2 (run management)

**Files:**
- Modify: `src/dashboard/notebook.py`

This task adds Start / Resume / Stop buttons to the `_Monitor` class. It is purely additive
— Phase 1 `monitor()` continues to work unchanged.

- [ ] **Step 1: Add run management methods to `_Monitor`**

Add the following imports at the top of `notebook.py` (after the existing imports):

```python
from src.dashboard.run_manager import launch_training, kill_training
```

Then add three attributes inside `_Monitor.__init__`, after `self._plot_w`:

```python
        # Phase 2: run management
        self._config_path = Path("configs/default.yaml")
        self._proc = None
        self._proc_lock = threading.Lock()

        self._start_btn = widgets.Button(description="▶ Start", button_style="success",
                                         layout=widgets.Layout(width="100px"))
        self._resume_btn = widgets.Button(description="⏩ Resume", button_style="info",
                                          layout=widgets.Layout(width="100px"))
        self._stop_btn2 = widgets.Button(description="⏹ Stop", button_style="danger",
                                         layout=widgets.Layout(width="100px"),
                                         disabled=True)
        self._action_out = widgets.HTML()

        self._start_btn.on_click(self._on_start)
        self._resume_btn.on_click(self._on_resume)
        self._stop_btn2.on_click(self._on_stop)
```

Add these three methods to `_Monitor`:

```python
    def _on_start(self, _):
        with self._proc_lock:
            self._proc = launch_training(config_path=self._config_path)
        self._action_out.value = f"<b>Started.</b> PID: {self._proc.pid}"
        self._start_btn.disabled = True
        self._resume_btn.disabled = True
        self._stop_btn2.disabled = False

    def _on_resume(self, _):
        run_dir = self._get_run_dir()
        if run_dir is None:
            self._action_out.value = "<span style='color:red'>No run selected.</span>"
            return
        with self._proc_lock:
            self._proc = launch_training(config_path=self._config_path,
                                         resume_run_dir=run_dir)
        self._action_out.value = f"<b>Resumed</b> {run_dir.name}. PID: {self._proc.pid}"
        self._start_btn.disabled = True
        self._resume_btn.disabled = True
        self._stop_btn2.disabled = False

    def _on_stop(self, _):
        with self._proc_lock:
            if self._proc is not None:
                kill_training(self._proc)
                self._proc = None
        self._action_out.value = "<b>Training stopped.</b>"
        self._start_btn.disabled = False
        self._resume_btn.disabled = False
        self._stop_btn2.disabled = True
```

- [ ] **Step 2: Update `widget()` to include the run management row**

Replace the existing `widget()` method with:

```python
    def widget(self) -> widgets.VBox:
        refresh_header = widgets.HBox([self._dropdown, self._stop_btn])
        mgmt_row = widgets.HBox([
            self._start_btn, self._resume_btn, self._stop_btn2, self._action_out
        ])
        return widgets.VBox([
            refresh_header,
            mgmt_row,
            self._metrics_w,
            self._gpu_w,
            self._alerts_w,
            self._plot_w,
        ])
```

- [ ] **Step 3: Verify it imports cleanly**

```bash
uv run python -c "from src.dashboard.notebook import monitor; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/notebook.py
git commit -m "feat(dashboard): add Jupyter widget Phase 2 — Start/Resume/Stop run management"
```

---

## Task 12: Rich TUI

**Files:**
- Create: `src/dashboard/tui.py`

- [ ] **Step 1: Create tui.py**

```python
# src/dashboard/tui.py
"""ParrotLLM Training Dashboard — Rich terminal UI.

Read-only. Always shows the latest run. Exit with Ctrl+C.

Launch: uv run main.py --stage dashboard --tui [--tui-refresh N]
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import rich.box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.dashboard.metrics_reader import read_metrics, is_metrics_stale
from src.dashboard.system_monitor import get_system_stats
from src.dashboard.problem_detector import detect_problems, Severity
from src.dashboard.run_manager import get_latest_run_dir

_SEVERITY_STYLE = {Severity.ERROR: "red", Severity.WARNING: "yellow", Severity.INFO: "blue"}


def _progress_panel(run_dir: Path) -> Panel:
    metrics = read_metrics(run_dir)
    if not metrics.steps:
        return Panel(Text("No training data yet. Start training first.", style="dim"),
                     title="Progress")
    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    max_steps = metrics.config.get("max_steps")

    lines = [Text(f"Run: {run_dir.name}", style="bold")]

    step_line = f"Step {step:,}"
    if max_steps:
        step_line += f" / {max_steps:,}  ({100.0 * step / max_steps:.1f}%)"
    lines.append(Text(step_line))

    row = f"Train {loss:.4f}"
    if metrics.val_losses:
        row += f"  │  Val {metrics.val_losses[-1]:.4f}"
    if metrics.val_ppls:
        row += f"  │  Val PPL {metrics.val_ppls[-1]:.1f}"
    row += f"  │  LR {lr:.2e}"
    lines.append(Text(row))

    extra = []
    if metrics.grad_norms:
        extra.append(f"Grad Norm {metrics.grad_norms[-1]:.3f}")
    if metrics.tokens_per_sec:
        extra.append(f"Tok/s {metrics.tokens_per_sec[-1]:,.0f}")
    if metrics.best_step:
        extra.append(f"Best Step {metrics.best_step:,}")
    if extra:
        lines.append(Text("  │  ".join(extra)))

    stale, age = is_metrics_stale(run_dir)
    if stale:
        lines.append(Text(f"⚠ Metrics not updated for {age}s — training may have stalled.",
                          style="yellow"))

    return Panel(Group(*lines), title="Progress")


def _alerts_panel(run_dir: Path) -> Panel:
    metrics = read_metrics(run_dir)
    alerts = detect_problems(metrics)
    if not alerts:
        return Panel(Text("✅  No problems detected", style="green"), title="Alerts")
    table = Table.grid(padding=(0, 2))
    for a in alerts:
        style = _SEVERITY_STYLE[a.severity]
        table.add_row(
            Text("●", style=style),
            Text(a.code, style=f"bold {style}"),
            Text(f"— {a.message}"),
        )
    return Panel(table, title="Alerts")


def _system_panel() -> Panel:
    stats = get_system_stats()
    header = Text(
        f"CPU {stats.cpu_percent:.1f}%  │  "
        f"RAM {stats.ram_used_gb:.1f} / {stats.ram_total_gb:.1f} GB"
    )
    if not stats.gpu_available:
        return Panel(Group(header, Text("GPU: N/A", style="dim")), title="System")

    table = Table(box=rich.box.SIMPLE, show_header=True, header_style="bold",
                  show_edge=False, pad_edge=False)
    table.add_column("GPU", justify="right", style="dim")
    table.add_column("Name")
    table.add_column("Mem Used", justify="right")
    table.add_column("Mem Total", justify="right")
    table.add_column("Util", justify="right")
    table.add_column("Temp", justify="right")

    for g in stats.gpus:
        util = f"{g.utilization_pct:.0f}%" if not math.isnan(g.utilization_pct) else "—"
        temp = f"{g.temperature_c:.0f}°C" if not math.isnan(g.temperature_c) else "—"
        table.add_row(str(g.index), g.name,
                      f"{g.mem_used_gb:.1f} GB", f"{g.mem_total_gb:.1f} GB",
                      util, temp)

    avg_util = (f"{stats.gpu_avg_utilization:.0f}%"
                if not math.isnan(stats.gpu_avg_utilization) else "—")
    table.add_row(
        "[bold]ALL[/bold]", "",
        f"[bold]{stats.gpu_total_used_gb:.1f} GB[/bold]",
        f"[bold]{stats.gpu_total_mem_gb:.1f} GB[/bold]",
        f"[bold]{avg_util}[/bold]", "—",
    )

    return Panel(Group(header, table), title="System")


def _build_layout(run_dir: Path):
    from rich.layout import Layout
    layout = Layout()
    layout.split_column(
        Layout(_progress_panel(run_dir), name="progress", size=7),
        Layout(_alerts_panel(run_dir), name="alerts", size=5),
        Layout(_system_panel(), name="system"),
    )
    return layout


def run_tui(runs_dir: Path, refresh: int = 5) -> None:
    """Run the terminal dashboard. Blocks until Ctrl+C."""
    console = Console()

    run_dir = get_latest_run_dir(runs_dir)
    if run_dir is None:
        console.print("[yellow]No runs found in runs/. Start training first.[/yellow]")
        return

    console.print(
        f"[dim]ParrotLLM TUI — {refresh}s refresh — Ctrl+C to exit — "
        f"showing: {run_dir.name}[/dim]"
    )

    with Live(_build_layout(run_dir), refresh_per_second=0.5, console=console) as live:
        try:
            while True:
                time.sleep(refresh)
                # Re-detect latest run each tick in case a new run started
                latest = get_latest_run_dir(runs_dir)
                if latest:
                    run_dir = latest
                live.update(_build_layout(run_dir))
        except KeyboardInterrupt:
            pass
```

- [ ] **Step 2: Smoke-test that it imports cleanly**

```bash
uv run python -c "from src.dashboard.tui import run_tui; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Verify the full test suite still passes**

```bash
uv run pytest tests/dashboard/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/tui.py
git commit -m "feat(dashboard): add Rich TUI — read-only terminal dashboard"
```

---

## Self-Review

**Spec coverage:**
- ✅ Folder structure `src/dashboard/` — Task 2
- ✅ SSH tunnel / browser-on-server / --open / --share flags — Task 9
- ✅ Gradio Live Monitor: alerts (hidden when empty), progress, ETA, refresh slider, GPU table, Download PDF, log tail, stale warning, empty state — Task 8
- ✅ Gradio Architecture tab — Task 8
- ✅ Gradio Run Manager: Start, Resume, Stop, status indicator — Task 8
- ✅ Jupyter Phase 1 read-only — Task 10
- ✅ Jupyter Phase 2 run management — Task 11
- ✅ Rich TUI read-only — Task 12
- ✅ Per-GPU stats (individual + combined ALL row) in all frontends — Tasks 4, 8, 10, 12
- ✅ pynvml for utilization % and temperature — Task 4
- ✅ Stale data warning (60s threshold) in all frontends — Tasks 3, 8, 10, 12
- ✅ Empty state in all frontends — Tasks 8, 10, 12
- ✅ Consistent metric labels — enforced in `_metrics_html`, `_fmt_progress`, `_progress_panel`
- ✅ kill_training() in run_manager — Task 6
- ✅ get_log_lines() tail of train.log — Task 6
- ✅ plots.py light theme — Task 7

**Type consistency check:**
- `read_metrics()` → `TrainingMetrics` used in Tasks 3, 5, 6, 7, 8, 10, 12 ✅
- `get_system_stats()` → `SystemStats` with `.gpus: list[GPUStats]` used in Tasks 4, 8, 10, 12 ✅
- `detect_problems()` → `list[Alert]` used in Tasks 5, 8, 10, 12 ✅
- `kill_training(proc)` takes a `Popen` — consistent in Tasks 6, 8, 11 ✅
- `get_log_lines(run_dir, n=20)` → `list[str]` used in Tasks 6, 8 ✅
- `is_metrics_stale(run_dir)` → `tuple[bool, int]` used in Tasks 3, 8, 10, 12 ✅
- `run_tui(runs_dir, refresh)` called from main.py with `args.tui_refresh` ✅
- `run_dashboard(runs_dir, config_path, share, open_browser)` called from main.py ✅
