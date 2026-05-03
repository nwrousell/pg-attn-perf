# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

vllm_homo = pd.read_csv("results/vllm_b16_homogeneous/summary.csv")
hf_homo = pd.read_csv("results/hf_homogeneous/summary.csv")
vllm_hetero = pd.read_csv("results/vllm_b16_heterogeneous/summary.csv")
hf_hetero = pd.read_csv("results/hf_heterogeneous/summary.csv")

# %%
# Plot 1: Grouped bar chart — Throughput by workload type and backend
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(2)
width = 0.3

vllm_tps = [vllm_homo["throughput_tok_per_s"].iloc[0],
            vllm_hetero["throughput_tok_per_s"].iloc[0]]
hf_tps = [hf_homo["throughput_tok_per_s"].iloc[0],
          hf_hetero["throughput_tok_per_s"].iloc[0]]

bars1 = ax.bar(x - width/2, vllm_tps, width, label="vLLM (PagedAttention)", color="tab:blue")
bars2 = ax.bar(x + width/2, hf_tps, width, label="HF (Contiguous KV Cache)", color="tab:orange")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Homogeneous", "Heterogeneous"])
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("H2: Throughput by Workload Heterogeneity")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/h2_throughput_grouped.png", dpi=150)
plt.show()

# %%
# Plot 2: Speedup factor — vLLM throughput / HF throughput
speedup_homo = vllm_tps[0] / hf_tps[0]
speedup_hetero = vllm_tps[1] / hf_tps[1]

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(["Homogeneous", "Heterogeneous"],
              [speedup_homo, speedup_hetero],
              color=["tab:green", "tab:red"], width=0.4)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")
ax.set_ylabel("Speedup (vLLM / HF)")
ax.set_title("H2: PagedAttention Speedup Over Contiguous KV Cache")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/h2_speedup.png", dpi=150)
plt.show()

# %%
# Plot 3: Memory utilization grouped bar chart
fig, ax = plt.subplots(figsize=(8, 5))

def mem_pct(nvml_path):
    df = pd.read_csv(nvml_path)
    return (df["mem_used_mb"].mean() / df["mem_total_mb"].iloc[0]) * 100

vllm_mem = [mem_pct("results/vllm_b16_homogeneous/nvml.csv"),
            mem_pct("results/vllm_b16_heterogeneous/nvml.csv")]
hf_mem = [mem_pct("results/hf_homogeneous/nvml.csv"),
          mem_pct("results/hf_heterogeneous/nvml.csv")]

bars1 = ax.bar(x - width/2, vllm_mem, width, label="vLLM (PagedAttention)", color="tab:blue")
bars2 = ax.bar(x + width/2, hf_mem, width, label="HF (Contiguous KV Cache)", color="tab:orange")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Homogeneous", "Heterogeneous"])
ax.set_ylabel("GPU Memory Utilization (%)")
ax.set_ylim(0, 100)
ax.set_title("H2: Memory Utilization by Workload Heterogeneity")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/h2_memory.png", dpi=150)
plt.show()

# %%
# Summary table
comparison = pd.concat([vllm_homo, hf_homo, vllm_hetero, hf_hetero], ignore_index=True)
cols = ["name", "backend", "throughput_tok_per_s", "total_time_s",
        "peak_gpu_mem_used_mb", "avg_power_w"]
comparison[[c for c in cols if c in comparison.columns]]
