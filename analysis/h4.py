# %%
"""
H4: block size vs throughput for homogeneous vLLM runs (short vs longer prompts).

Research angle: for many short sequences, smaller block sizes can reduce KV
page waste from partially filled last blocks (roughly ~half full in expectation),
so more logical capacity / less internal fragmentation — potentially higher
throughput if batching is memory-bound.

Data: results/h4_* (e.g. h4_vllm_homogeneous_l16_b16) with summary.csv per run.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_ROOT = Path("results")
# Match experiment dirs (e.g. h4_vllm_homogeneous_l16_b16)
H4_DIR_PATTERN = re.compile(r"l(\d+)_b(\d+)$")


def load_h4_summaries(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("h4_"):
            continue
        m = H4_DIR_PATTERN.search(d.name)
        if not m:
            continue
        prompt_len = int(m.group(1))
        block_from_name = int(m.group(2))
        summary_path = d / "summary.csv"
        if not summary_path.is_file():
            continue
        df = pd.read_csv(summary_path)
        if len(df) != 1:
            raise ValueError(f"Expected one row in {summary_path}, got {len(df)}")
        row = df.iloc[0].to_dict()
        row["run_dir"] = d.name
        row["prompt_len_cfg"] = prompt_len
        if int(row["block_size"]) != block_from_name:
            raise ValueError(
                f"block_size mismatch: dir {d.name} vs summary {row['block_size']}"
            )
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No h4_* runs under {results_root.resolve()} (need summary.csv)."
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["prompt_len_cfg", "block_size"]).reset_index(drop=True)
    return out


# %%
summary = load_h4_summaries()
T = (
    (summary["total_prompt_tokens"] + summary["total_output_tokens"])
    / summary["num_sequences"]
).astype(int)
B = summary["block_size"].astype(int)
# Wasted token slots in the final partial KV block (0 if perfectly aligned).
summary["avg_seq_tokens"] = T
summary["last_block_slack_tokens"] = (-T % B)
summary

# %%
# Throughput: block size 16 vs 32, faceted by configured prompt length
prompt_lengths = sorted(summary["prompt_len_cfg"].unique())
n = len(prompt_lengths)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
if n == 1:
    axes = [axes]

for ax, L in zip(axes, prompt_lengths):
    sub = summary[summary["prompt_len_cfg"] == L].sort_values("block_size")
    xs = sub["block_size"].astype(str).tolist()
    ys = sub["throughput_tok_per_s"].tolist()
    bars = ax.bar(xs, ys, color=["tab:green", "tab:purple"][: len(xs)], width=0.55)
    for bar, val in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("KV block size")
    ax.set_title(f"Prompt length L = {L}")
    ax.grid(True, alpha=0.3, axis="y")

axes[0].set_ylabel("Throughput (tok/s)")
fig.suptitle(
    "H4: Throughput vs block size (homogeneous batch, fixed decode length)",
    y=1.02,
)
plt.tight_layout()
out_throughput = RESULTS_ROOT / "h4_throughput_by_block.png"
plt.savefig(out_throughput, dpi=150, bbox_inches="tight")
plt.show()

# %%
# Peak / average GPU memory from summary (proxy for allocator pressure)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, col, title in zip(
    axes,
    ["peak_gpu_mem_used_mb", "avg_gpu_mem_used_mb"],
    ["Peak GPU memory (MB)", "Avg GPU memory (MB)"],
):
    for L in prompt_lengths:
        sub = summary[summary["prompt_len_cfg"] == L].sort_values("block_size")
        ax.plot(
            sub["block_size"],
            sub[col],
            marker="o",
            label=f"L = {L}",
        )
    ax.set_xlabel("KV block size")
    ax.set_ylabel(title.split("(")[0].strip())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("H4: Reported GPU memory vs block size", y=1.02)
plt.tight_layout()
out_mem = RESULTS_ROOT / "h4_memory_by_block.png"
plt.savefig(out_mem, dpi=150, bbox_inches="tight")
plt.show()

# %%
# Table: b16 vs b32 relative throughput and memory per prompt length
def pct_delta(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")


pivot_tp = summary.pivot_table(
    index="prompt_len_cfg",
    columns="block_size",
    values="throughput_tok_per_s",
)
pivot_peak = summary.pivot_table(
    index="prompt_len_cfg",
    columns="block_size",
    values="peak_gpu_mem_used_mb",
)
pivot_slack = summary.pivot_table(
    index="prompt_len_cfg",
    columns="block_size",
    values="last_block_slack_tokens",
)

comparison = pd.DataFrame(
    {
        "throughput_b16": pivot_tp[16],
        "throughput_b32": pivot_tp[32],
        "tp_pct_b16_vs_b32": [
            pct_delta(pivot_tp.loc[L, 16], pivot_tp.loc[L, 32])
            for L in pivot_tp.index
        ],
        "peak_mem_mb_b16": pivot_peak[16],
        "peak_mem_mb_b32": pivot_peak[32],
        "last_block_slack_b16": pivot_slack[16],
        "last_block_slack_b32": pivot_slack[32],
    }
)
comparison.index.name = "prompt_len_cfg"
comparison = comparison.reset_index()
comparison

# %%
# Optional: NVML time series from first run dir per (L, block) for spot checks
# (uncomment if you want raw traces)
# ex = RESULTS_ROOT / sorted(RESULTS_ROOT.glob("h4_*"))[0] / "nvml.csv"
# pd.read_csv(ex).head()
