from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VLLM_DIR = Path("results/vllm_b16_memory_pressure")
HF_DIR = Path("results/hf_memory_pressure")
OUT_DIR = Path("results/h5_figures")


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_scalar(df: pd.DataFrame, col: str, default=float("nan")):
    return df[col].iloc[0] if col in df.columns and not df.empty else default


def energy_from_nvml(nvml_df: pd.DataFrame) -> float:
    """Estimate energy in Joules from sampled power."""
    if nvml_df.empty or "timestamp_s" not in nvml_df.columns or "power_w" not in nvml_df.columns:
        return float("nan")
    if len(nvml_df) < 2:
        return float("nan")
    dt = nvml_df["timestamp_s"].diff().fillna(0.0)
    return float((nvml_df["power_w"] * dt).sum())


def avg_gpu_mem_used(summary_df: pd.DataFrame, nvml_df: pd.DataFrame) -> float:
    if "avg_gpu_mem_used_mb" in summary_df.columns:
        return summary_df["avg_gpu_mem_used_mb"].iloc[0]
    if "mem_used_mb" in nvml_df.columns:
        return float(nvml_df["mem_used_mb"].mean())
    return float("nan")


def gpu_mem_total(summary_df: pd.DataFrame, nvml_df: pd.DataFrame) -> float:
    if "gpu_mem_total_mb" in summary_df.columns:
        return summary_df["gpu_mem_total_mb"].iloc[0]
    if "mem_total_mb" in nvml_df.columns:
        return nvml_df["mem_total_mb"].iloc[0]
    return float("nan")


def peak_gpu_mem_used(summary_df: pd.DataFrame, nvml_df: pd.DataFrame) -> float:
    if "peak_gpu_mem_used_mb" in summary_df.columns:
        return summary_df["peak_gpu_mem_used_mb"].iloc[0]
    if "mem_used_mb" in nvml_df.columns:
        return float(nvml_df["mem_used_mb"].max())
    return float("nan")


def fragmentation(summary_df: pd.DataFrame) -> float:
    if "mem_fragmentation_mb" in summary_df.columns:
        return summary_df["mem_fragmentation_mb"].iloc[0]
    if "peak_mem_reserved_mb" in summary_df.columns and "peak_mem_allocated_mb" in summary_df.columns:
        return summary_df["peak_mem_reserved_mb"].iloc[0] - summary_df["peak_mem_allocated_mb"].iloc[0]
    return float("nan")


def mem_gb(summary_df: pd.DataFrame, nvml_df: pd.DataFrame) -> float:
    total_mb = gpu_mem_total(summary_df, nvml_df)
    if pd.isna(total_mb):
        return float("nan")
    return total_mb / 1024.0


def label_for(summary_df: pd.DataFrame, nvml_df: pd.DataFrame) -> str:
    backend = get_scalar(summary_df, "backend", "")
    gpu_gb = mem_gb(summary_df, nvml_df)
    gpu_suffix = f", {gpu_gb:.0f} GB GPU" if pd.notna(gpu_gb) else ""
    if backend == "vllm":
        block_size = get_scalar(summary_df, "block_size", float("nan"))
        if pd.notna(block_size):
            return f"vLLM\n(PagedAttention, b={int(block_size)}{gpu_suffix})"
        return f"vLLM\n(PagedAttention{gpu_suffix})"
    return f"HF\n(Static Cache{gpu_suffix})"


def bar_plot(labels, values, ylabel, title, path, colors, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    finite_values = [v for v in values if pd.notna(v)]
    ymax = max(finite_values) if finite_values else 1.0
    for bar, val in zip(bars, values):
        if pd.isna(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ymax * 0.02, 0.01),
            fmt.format(val),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def timeline_plot(vllm_nvml, hf_nvml, column, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    if column in vllm_nvml.columns:
        ax.plot(vllm_nvml["timestamp_s"], vllm_nvml[column], label="vLLM", linewidth=2.1, color="tab:blue")
    if column in hf_nvml.columns:
        ax.plot(hf_nvml["timestamp_s"], hf_nvml[column], label="HF", linewidth=2.1, color="tab:orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def utilization_plot(vllm_summary, hf_summary, vllm_nvml, hf_nvml, labels, colors):
    total_mem = [
        gpu_mem_total(vllm_summary, vllm_nvml),
        gpu_mem_total(hf_summary, hf_nvml),
    ]
    avg_used = [
        avg_gpu_mem_used(vllm_summary, vllm_nvml),
        avg_gpu_mem_used(hf_summary, hf_nvml),
    ]
    util_pct = [
        100 * used / total if pd.notna(used) and pd.notna(total) and total > 0 else float("nan")
        for used, total in zip(avg_used, total_mem)
    ]
    bar_plot(
        labels,
        util_pct,
        "Average GPU Memory Utilization (%)",
        "H5: Average GPU Memory Utilization",
        OUT_DIR / "h5_avg_gpu_mem_utilization.png",
        colors,
        "{:.1f}%",
    )


def request_length_plot(vllm_requests, hf_requests):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(vllm_requests["output_tokens"], bins=25, alpha=0.7, label="vLLM", color="tab:blue")
    axes[0].hist(hf_requests["output_tokens"], bins=25, alpha=0.7, label="HF", color="tab:orange")
    axes[0].set_title("Output Token Distribution")
    axes[0].set_xlabel("Generated Tokens")
    axes[0].set_ylabel("Request Count")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].scatter(
        vllm_requests["prompt_tokens"],
        vllm_requests["output_tokens"],
        s=10,
        alpha=0.45,
        label="vLLM",
        color="tab:blue",
    )
    axes[1].scatter(
        hf_requests["prompt_tokens"],
        hf_requests["output_tokens"],
        s=10,
        alpha=0.45,
        label="HF",
        color="tab:orange",
    )
    axes[1].set_title("Prompt Length vs Output Length")
    axes[1].set_xlabel("Prompt Tokens")
    axes[1].set_ylabel("Generated Tokens")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "h5_request_shapes.png", dpi=180)
    plt.close(fig)


def relative_performance_plot(labels, throughput, total_mem_gb, colors):
    tok_per_gb = [
        t / m if pd.notna(t) and pd.notna(m) and m > 0 else float("nan")
        for t, m in zip(throughput, total_mem_gb)
    ]
    bar_plot(
        labels,
        tok_per_gb,
        "Throughput / GPU Memory (tok/s/GB)",
        "H5: Throughput Normalized by GPU Memory Capacity",
        OUT_DIR / "h5_throughput_per_gb.png",
        colors,
        "{:.1f}",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vllm_summary = load_csv(VLLM_DIR, "summary.csv")
    hf_summary = load_csv(HF_DIR, "summary.csv")
    vllm_nvml = load_csv(VLLM_DIR, "nvml.csv")
    hf_nvml = load_csv(HF_DIR, "nvml.csv")
    vllm_requests = load_csv(VLLM_DIR, "requests.csv")
    hf_requests = load_csv(HF_DIR, "requests.csv")

    labels = [label_for(vllm_summary, vllm_nvml), label_for(hf_summary, hf_nvml)]
    colors = ["tab:blue", "tab:orange"]
    total_mem_gb = [mem_gb(vllm_summary, vllm_nvml), mem_gb(hf_summary, hf_nvml)]

    throughput = [
        get_scalar(vllm_summary, "throughput_tok_per_s"),
        get_scalar(hf_summary, "throughput_tok_per_s"),
    ]
    bar_plot(
        labels,
        throughput,
        "Throughput (tok/s)",
        "H5: Cross-GPU Throughput Under Memory Pressure",
        OUT_DIR / "h5_throughput.png",
        colors,
        "{:.1f}",
    )
    relative_performance_plot(labels, throughput, total_mem_gb, colors)

    total_time = [
        get_scalar(vllm_summary, "total_time_s"),
        get_scalar(hf_summary, "total_time_s"),
    ]
    bar_plot(
        labels,
        total_time,
        "Completion Time (s)",
        "H5: Cross-GPU End-to-End Runtime",
        OUT_DIR / "h5_total_time.png",
        colors,
        "{:.2f}",
    )

    peak_mem = [
        peak_gpu_mem_used(vllm_summary, vllm_nvml),
        peak_gpu_mem_used(hf_summary, hf_nvml),
    ]
    bar_plot(
        labels,
        peak_mem,
        "Peak GPU Memory Used (MB)",
        "H5: Cross-GPU Peak Memory Used",
        OUT_DIR / "h5_peak_gpu_mem_used.png",
        colors,
        "{:.0f}",
    )

    utilization_plot(vllm_summary, hf_summary, vllm_nvml, hf_nvml, labels, colors)

    avg_power = [
        get_scalar(vllm_summary, "avg_power_w", float(vllm_nvml["power_w"].mean())),
        get_scalar(hf_summary, "avg_power_w", float(hf_nvml["power_w"].mean())),
    ]
    bar_plot(
        labels,
        avg_power,
        "Average Power (W)",
        "H5: Cross-GPU Average Power",
        OUT_DIR / "h5_avg_power.png",
        colors,
        "{:.1f}",
    )

    energy_proxy = [energy_from_nvml(vllm_nvml), energy_from_nvml(hf_nvml)]
    bar_plot(
        labels,
        energy_proxy,
        "Estimated Energy (J)",
        "H5: Cross-GPU Energy Proxy from NVML Samples",
        OUT_DIR / "h5_energy_proxy.png",
        colors,
        "{:.0f}",
    )

    frag = [fragmentation(vllm_summary), fragmentation(hf_summary)]
    if any(pd.notna(v) for v in frag):
        bar_plot(
            labels,
            frag,
            "Reserved - Allocated (MB)",
            "H5: CUDA Memory Fragmentation Estimate",
            OUT_DIR / "h5_fragmentation.png",
            colors,
            "{:.0f}",
        )

    timeline_plot(
        vllm_nvml,
        hf_nvml,
        "mem_used_mb",
        "GPU Memory Used (MB)",
        "H5: Cross-GPU Memory Over Time",
        OUT_DIR / "h5_memory_timeline.png",
    )
    timeline_plot(
        vllm_nvml,
        hf_nvml,
        "power_w",
        "Power (W)",
        "H5: Cross-GPU Power Over Time",
        OUT_DIR / "h5_power_timeline.png",
    )
    timeline_plot(
        vllm_nvml,
        hf_nvml,
        "gpu_util_pct",
        "GPU Utilization (%)",
        "H5: Cross-GPU Utilization Over Time",
        OUT_DIR / "h5_gpu_util_timeline.png",
    )

    request_length_plot(vllm_requests, hf_requests)

    comparison = pd.DataFrame(
        [
            {
                "backend": "vllm",
                "throughput_tok_per_s": throughput[0],
                "total_time_s": total_time[0],
                "peak_gpu_mem_used_mb": peak_mem[0],
                "avg_gpu_mem_used_mb": avg_gpu_mem_used(vllm_summary, vllm_nvml),
                "gpu_mem_total_mb": gpu_mem_total(vllm_summary, vllm_nvml),
                "gpu_mem_total_gb": total_mem_gb[0],
                "avg_power_w": avg_power[0],
                "energy_j_estimate": energy_proxy[0],
                "mem_fragmentation_mb": frag[0],
            },
            {
                "backend": "hf",
                "throughput_tok_per_s": throughput[1],
                "total_time_s": total_time[1],
                "peak_gpu_mem_used_mb": peak_mem[1],
                "avg_gpu_mem_used_mb": avg_gpu_mem_used(hf_summary, hf_nvml),
                "gpu_mem_total_mb": gpu_mem_total(hf_summary, hf_nvml),
                "gpu_mem_total_gb": total_mem_gb[1],
                "avg_power_w": avg_power[1],
                "energy_j_estimate": energy_proxy[1],
                "mem_fragmentation_mb": frag[1],
            },
        ]
    )
    if pd.notna(throughput[1]) and throughput[1] != 0:
        comparison["throughput_speedup_vs_hf"] = comparison["throughput_tok_per_s"] / throughput[1]
    comparison["throughput_tok_per_s_per_gb"] = comparison["throughput_tok_per_s"] / comparison["gpu_mem_total_gb"]
    comparison["peak_mem_fraction"] = comparison["peak_gpu_mem_used_mb"] / comparison["gpu_mem_total_mb"]
    comparison.to_csv(OUT_DIR / "h5_comparison.csv", index=False)

    print(f"Saved H5 figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
