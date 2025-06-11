"""Combined metrics module."""
from typing import Dict, Any, Tuple, Optional, List
import logging
import time

from .base_metric import BaseMetric

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


class GPUMetric(BaseMetric):
    """Collect GPU utilization and memory stats using NVML."""

    def __init__(self) -> None:
        super().__init__()
        self.gpu_handle = None
        self.init_nvml()

    def init_nvml(self) -> None:
        if not pynvml:
            logging.warning("pynvml module not found. GPU metrics will be reported as 0.")
            return
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logging.info("NVML initialized for GPU monitoring")
        except pynvml.NVMLError as e:
            logging.warning(f"Cannot initialize NVML: {e}")
            logging.warning(
                "GPU metrics will be reported as 0. If you have a compatible GPU, make sure drivers are installed correctly.")
            self.gpu_handle = None

    def shutdown(self) -> None:
        if self.gpu_handle and pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logging.warning(f"Error shutting down NVML: {e}")
            self.gpu_handle = None

    def get_gpu_utilization(self) -> float:
        if not self.gpu_handle or not pynvml:
            return 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return util.gpu
        except pynvml.NVMLError as e:
            logging.warning(f"Error retrieving GPU utilization: {e}")
            return 0.0

    def get_memory_usage(self) -> Tuple[float, float]:
        if not self.gpu_handle or not pynvml:
            return 0.0, 0.0
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_mb = info.used / (1024 * 1024)
            total_mb = info.total / (1024 * 1024)
            return used_mb, total_mb
        except pynvml.NVMLError as e:
            logging.warning(f"Error retrieving GPU memory usage: {e}")
            return 0.0, 0.0

    def collect(self, **kwargs) -> Dict[str, Any]:
        if not self.gpu_handle and kwargs.get('attempt_reinit', False):
            self.init_nvml()

        utilization = self.get_gpu_utilization()
        memory_used, memory_total = self.get_memory_usage()
        memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0

        self.values.append({'utilization': utilization, 'memory_percent': memory_percent})

        return {
            'gpu_utilization': utilization,
            'gpu_memory_used_mb': memory_used,
            'gpu_memory_total_mb': memory_total,
            'gpu_memory_percent': memory_percent,
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.values:
            return {
                'avg_utilization': 0.0,
                'avg_memory_percent': 0.0,
                'count': 0,
            }
        avg_util = sum(v['utilization'] for v in self.values) / len(self.values)
        avg_mem = sum(v['memory_percent'] for v in self.values) / len(self.values)
        return {
            'avg_utilization': avg_util,
            'avg_memory_percent': avg_mem,
            'count': len(self.values),
        }


class MemoryMetric(BaseMetric):
    """Collect system RAM usage metrics."""

    def get_ram_usage(self) -> Tuple[float, float, float]:
        if not psutil:
            logging.warning("psutil module not found. Memory metrics will be zeros.")
            return 0.0, 0.0, 0.0
        try:
            ram = psutil.virtual_memory()
            used_mb = ram.used / (1024 * 1024)
            total_mb = ram.total / (1024 * 1024)
            percent = ram.percent
            return used_mb, total_mb, percent
        except Exception as e:
            logging.warning(f"Error retrieving RAM usage: {e}")
            return 0.0, 0.0, 0.0

    def collect(self, **kwargs) -> Dict[str, Any]:
        used_mb, total_mb, percent = self.get_ram_usage()
        self.values.append(percent)
        return {
            'ram_used_mb': used_mb,
            'ram_total_mb': total_mb,
            'ram_percent': percent,
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.values:
            return {
                'avg_ram_percent': 0.0,
                'peak_ram_percent': 0.0,
                'count': 0,
            }
        avg_ram = sum(self.values) / len(self.values)
        peak_ram = max(self.values)
        return {
            'avg_ram_percent': avg_ram,
            'peak_ram_percent': peak_ram,
            'count': len(self.values),
        }


class PowerMetric(BaseMetric):
    """Collect GPU power usage and estimate energy consumption."""

    def __init__(self) -> None:
        super().__init__()
        self.gpu_handle = None
        self.start_time = time.time()
        self.total_energy_joules = 0.0
        self.init_nvml()

    def init_nvml(self) -> None:
        if not pynvml:
            logging.warning("pynvml module not found. Power metrics will be 0.")
            return
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logging.info("NVML initialized for power monitoring")
        except pynvml.NVMLError as e:
            logging.warning(f"Cannot initialize NVML for power monitoring: {e}")
            logging.warning(
                "Power metrics will be reported as 0. If you have a compatible GPU, make sure drivers are installed correctly.")
            self.gpu_handle = None

    def shutdown(self) -> None:
        if self.gpu_handle and pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logging.warning(f"Error shutting down NVML power monitoring: {e}")
            self.gpu_handle = None

    def get_gpu_power_usage(self) -> float:
        if not self.gpu_handle or not pynvml:
            return 0.0
        try:
            power_usage = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            return power_usage
        except pynvml.NVMLError as e:
            logging.warning(f"Error retrieving GPU power usage: {e}")
            return 0.0

    def collect(self, **kwargs) -> Dict[str, Any]:
        if not self.gpu_handle and kwargs.get('attempt_reinit', False):
            self.init_nvml()

        power_usage = self.get_gpu_power_usage()
        interval = kwargs.get('interval', None)
        energy_joules = 0.0
        if interval is not None and interval > 0:
            energy_joules = power_usage * interval
            self.total_energy_joules += energy_joules
        self.values.append(power_usage)
        return {
            'power_usage_watts': power_usage,
            'energy_joules': energy_joules,
            'total_energy_joules': self.total_energy_joules,
            'total_energy_wh': self.total_energy_joules / 3600.0,
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.values:
            return {
                'avg_power_watts': 0.0,
                'peak_power_watts': 0.0,
                'total_energy_wh': 0.0,
                'collection_duration_s': 0.0,
            }
        avg_power = sum(self.values) / len(self.values)
        peak_power = max(self.values)
        duration = time.time() - self.start_time
        return {
            'avg_power_watts': avg_power,
            'peak_power_watts': peak_power,
            'total_energy_wh': self.total_energy_joules / 3600.0,
            'collection_duration_s': duration,
        }


__all__ = [
    'GPUMetric',
    'MemoryMetric',
    'PowerMetric',
]
