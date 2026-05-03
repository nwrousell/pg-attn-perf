"""
Experiment runner.

Usage: python runner.py configs/example.yaml
"""

import os
import sys
import time

import pandas as pd
import yaml
from dotenv import load_dotenv

from backends import create_backend
from stats import NVMLMonitor, VLLMMetricsMonitor
from workload import generate_workload

load_dotenv() # for HF token (to access gated models)

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(config: dict):
    name = config["name"]
    out_dir = os.path.join("results", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Experiment: {name} ===")
    print(f"  Backend: {config['backend']}")
    print(f"  Model:   {config['model']}")
    if config["backend"] == "vllm":
        print(f"  Block size: {config.get('block_size', 16)}")

    # Set up backend (loads model)
    print("Loading model...")
    backend = create_backend(config)
    tokenizer = backend.get_tokenizer()

    # Generate workload
    print("Generating workload...")
    wl = generate_workload(config, tokenizer)
    print(f"  {len(wl)} sequences, "
          f"prompt tokens: {min(len(tokenizer.encode(w['prompt'])) for w in wl)}"
          f"-{max(len(tokenizer.encode(w['prompt'])) for w in wl)}, "
          f"output tokens: {min(w['max_tokens'] for w in wl)}"
          f"-{max(w['max_tokens'] for w in wl)}")

    # Set up monitoring
    stats_cfg = config.get("stats", {})
    nvml_interval = stats_cfg.get("nvml_poll_interval", 0.1)
    monitor = NVMLMonitor(poll_interval=nvml_interval)

    vllm_monitor = None
    if config["backend"] == "vllm":
        vllm_poll = stats_cfg.get("vllm_metrics_poll_interval", 0.5)
        vllm_monitor = VLLMMetricsMonitor(poll_interval=vllm_poll)

    # Run
    print("Running workload...")
    monitor.start()
    if vllm_monitor:
        vllm_monitor.start()
    t0 = time.perf_counter()

    request_df = backend.run(wl)

    total_time = time.perf_counter() - t0
    monitor.stop()
    if vllm_monitor:
        vllm_monitor.stop()

    # Build summary
    total_output_tokens = request_df["output_tokens"].sum()
    total_prompt_tokens = request_df["prompt_tokens"].sum()
    nvml_df = monitor.to_dataframe()

    summary = {
        "name": name,
        "backend": config["backend"],
        "model": config["model"],
        "block_size": config.get("block_size", None),
        "num_sequences": len(wl),
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_output_tokens": int(total_output_tokens),
        "total_time_s": total_time,
        "throughput_tok_per_s": total_output_tokens / total_time,
        "avg_latency_s": request_df["latency_s"].mean(),
    }

    if not nvml_df.empty:
        summary["avg_gpu_util_pct"] = nvml_df["gpu_util_pct"].mean()
        summary["peak_gpu_mem_used_mb"] = nvml_df["mem_used_mb"].max()
        summary["avg_gpu_mem_used_mb"] = nvml_df["mem_used_mb"].mean()
        summary["gpu_mem_total_mb"] = nvml_df["mem_total_mb"].iloc[0]
        summary["avg_power_w"] = nvml_df["power_w"].mean()

    vllm_metrics_df = pd.DataFrame()
    if vllm_monitor:
        vllm_metrics_df = vllm_monitor.to_dataframe()
        summary.update(vllm_monitor.get_summary())

    # Write outputs
    request_df.to_csv(os.path.join(out_dir, "requests.csv"), index=False)
    nvml_df.to_csv(os.path.join(out_dir, "nvml.csv"), index=False)
    if not vllm_metrics_df.empty:
        vllm_metrics_df.to_csv(os.path.join(out_dir, "vllm_metrics.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"\nResults written to {out_dir}/")
    print(f"  Throughput: {summary['throughput_tok_per_s']:.1f} tok/s")
    print(f"  Total time: {total_time:.2f}s")
    if not nvml_df.empty:
        print(f"  Peak GPU mem: {summary['peak_gpu_mem_used_mb']:.0f} MB "
              f"/ {summary['gpu_mem_total_mb']:.0f} MB")
        print(f"  Avg power:    {summary['avg_power_w']:.1f} W")
    if vllm_monitor and "peak_gpu_cache_usage_pct" in summary:
        print(f"  Peak KV cache usage: {summary['peak_gpu_cache_usage_pct']:.1f}%")
        print(f"  Preemptions:         {summary.get('total_preemptions', 0)}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_experiment(config)


if __name__ == "__main__":
    main()
