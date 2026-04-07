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
