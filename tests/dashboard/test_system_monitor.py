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
