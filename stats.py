import threading
import time

import pandas as pd
import pynvml
import torch


class NVMLMonitor:
    """Background thread that polls GPU stats via NVML."""

    def __init__(self, poll_interval: float = 0.1, gpu_index: int = 0):
        self.poll_interval = poll_interval
        self.gpu_index = gpu_index
        self._readings: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self.poll_interval <= 0:
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._stop_event.clear()
        self._readings.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._start_time = time.monotonic()
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        pynvml.nvmlShutdown()

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )

                self._readings.append({
                    "timestamp_s": time.monotonic() - self._start_time,
                    "gpu_util_pct": util.gpu,
                    "mem_used_mb": mem_info.used / (1024 ** 2),
                    "mem_total_mb": mem_info.total / (1024 ** 2),
                    "mem_util_pct": util.memory,
                    "power_w": power,
                    "temp_c": temp,
                })
            except pynvml.NVMLError:
                pass
            self._stop_event.wait(self.poll_interval)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._readings:
            return pd.DataFrame(columns=[
                "timestamp_s", "gpu_util_pct", "mem_used_mb", "mem_total_mb",
                "mem_util_pct", "power_w", "temp_c",
            ])
        return pd.DataFrame(self._readings)


class VLLMMetricsMonitor:
    """Background thread that polls vLLM's Prometheus metrics from the default registry."""

    GAUGE_METRICS = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_swapped",
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:num_preemptions_total",
    ]

    def __init__(self, poll_interval: float = 0.5):
        self.poll_interval = poll_interval
        self._readings: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self.poll_interval <= 0:
            return
        self._stop_event.clear()
        self._readings.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _poll_loop(self):
        try:
            from prometheus_client import REGISTRY
        except ImportError:
            return

        while not self._stop_event.is_set():
            reading = {"timestamp_s": time.monotonic() - self._start_time}

            for metric in REGISTRY.collect():
                name = metric.name
                if name not in self.GAUGE_METRICS:
                    continue
                for sample in metric.samples:
                    if sample.labels:
                        continue
                    col = name.replace("vllm:", "").replace(":", "_")
                    reading[col] = sample.value

            if len(reading) > 1:
                self._readings.append(reading)

            self._stop_event.wait(self.poll_interval)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._readings:
            cols = ["timestamp_s"] + [
                m.replace("vllm:", "").replace(":", "_")
                for m in self.GAUGE_METRICS
            ]
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self._readings)

    def get_summary(self) -> dict:
        """Extract summary stats from the final snapshot and time-series."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        summary = {}
        if "gpu_cache_usage_perc" in df.columns:
            summary["avg_gpu_cache_usage_pct"] = df["gpu_cache_usage_perc"].mean() * 100
            summary["peak_gpu_cache_usage_pct"] = df["gpu_cache_usage_perc"].max() * 100
        if "num_preemptions_total" in df.columns:
            summary["total_preemptions"] = int(df["num_preemptions_total"].iloc[-1])
        if "avg_generation_throughput_toks_per_s" in df.columns:
            valid = df["avg_generation_throughput_toks_per_s"].dropna()
            if not valid.empty:
                summary["vllm_avg_gen_throughput"] = valid.mean()
        return summary


def reset_cuda_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_cuda_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return {
        "peak_mem_allocated_mb": peak_allocated,
        "peak_mem_reserved_mb": peak_reserved,
        "mem_fragmentation_mb": peak_reserved - peak_allocated,
    }
