# %%
import pandas as pd
import matplotlib.pyplot as plt

VLLM_DIR = "results/vllm_b16_homogeneous"
HF_DIR = "results/hf_homogeneous"

# %%
vllm_summary = pd.read_csv(f"{VLLM_DIR}/summary.csv")
hf_summary = pd.read_csv(f"{HF_DIR}/summary.csv")

vllm_nvml = pd.read_csv(f"{VLLM_DIR}/nvml.csv")
hf_nvml = pd.read_csv(f"{HF_DIR}/nvml.csv")

# %%
# Plot 1: Total throughput bar chart
fig, ax = plt.subplots(figsize=(6, 5))

labels = ["vLLM\n(PagedAttention)", "HF\n(Contiguous KV Cache)"]
throughputs = [
    vllm_summary["throughput_tok_per_s"].iloc[0],
    hf_summary["throughput_tok_per_s"].iloc[0],
]
bars = ax.bar(labels, throughputs, color=["tab:blue", "tab:orange"], width=0.5)

for bar, val in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.0f}", ha="center", va="bottom", fontweight="bold")

ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Total Throughput: vLLM vs HF (Homogeneous)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/h1_throughput.png", dpi=150)
plt.show()

# %%
# Plot 2: Memory utilization (% of total GPU memory used)
vllm_mem_pct = (vllm_nvml["mem_used_mb"].mean() / vllm_nvml["mem_total_mb"].iloc[0]) * 100
hf_mem_pct = (hf_nvml["mem_used_mb"].mean() / hf_nvml["mem_total_mb"].iloc[0]) * 100

fig, ax = plt.subplots(figsize=(6, 5))

labels = ["vLLM\n(PagedAttention)", "HF\n(Contiguous KV Cache)"]
mem_pcts = [vllm_mem_pct, hf_mem_pct]
bars = ax.bar(labels, mem_pcts, color=["tab:blue", "tab:orange"], width=0.5)

for bar, val in zip(bars, mem_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

ax.set_ylabel("GPU Memory Utilization (%)")
ax.set_ylim(0, 100)
ax.set_title("Memory Utilization: vLLM vs HF (Homogeneous)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/h1_memory.png", dpi=150)
plt.show()

# %%
# Summary table
comparison = pd.concat([vllm_summary, hf_summary], ignore_index=True)
cols = ["name", "backend", "throughput_tok_per_s", "total_time_s",
        "peak_gpu_mem_used_mb", "avg_power_w"]
comparison[[c for c in cols if c in comparison.columns]]
