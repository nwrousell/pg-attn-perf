# %%
"""
H4: block size vs throughput for homogeneous vLLM runs (short vs longer prompts).

Research angle: for many short sequences, smaller KV block sizes can reduce
waste in the last partially used logical block (internal fragmentation of
logical capacity), which might improve throughput if batching is memory-bound.

Data: results/h4_* (e.g. h4_vllm_homogeneous_l16_b16) with summary.csv per run.

Metrics notes:
- `avg_seq_tokens` / `last_block_slack_tokens` use (total_prompt_tokens +
  total_output_tokens) / num_sequences from the benchmark summary. That matches
  total generated token accounting in the workload CSVs but is *not* a full
  model of vLLM physical KV layout (prefix caching, alignment, metadata).
- Throughput / latency here are end-to-end for the whole benchmark window.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"

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


def validate_h4_design_grid(summary: pd.DataFrame) -> None:
    """Warn if the (prompt_len_cfg × block_size) grid is incomplete."""
    lengths = sorted(summary["prompt_len_cfg"].unique())
    blocks = sorted(summary["block_size"].unique())
    missing: list[tuple[int, int]] = []
    for L in lengths:
        for b in blocks:
            if summary[
                (summary["prompt_len_cfg"] == L) & (summary["block_size"] == b)
            ].empty:
                missing.append((L, b))
    if missing:
        warnings.warn(
            "Incomplete H4 factorial grid (missing runs): "
            + ", ".join(f"L={L}, block={b}" for L, b in missing),
            stacklevel=2,
        )


def attach_request_latency_spread(summary: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    """Per-run spread of per-sequence latency from requests.csv (often zero std in batched mode)."""
    means, stds, ns = [], [], []
    for _, r in summary.iterrows():
        p = results_root / r["run_dir"] / "requests.csv"
        if not p.is_file():
            means.append(float("nan"))
            stds.append(float("nan"))
            ns.append(0)
            continue
        req = pd.read_csv(p)
        if "latency_s" not in req.columns:
            means.append(float("nan"))
            stds.append(float("nan"))
            ns.append(0)
            continue
        lat = req["latency_s"].astype(float)
        means.append(float(lat.mean()))
        stds.append(float(lat.std(ddof=0)))
        ns.append(int(len(lat)))
    out = summary.copy()
    out["req_latency_mean_s"] = means
    out["req_latency_std_s"] = stds
    out["req_n"] = ns
    return out


def build_h4_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot throughput / memory / slack by prompt length and block size, plus
    relative deltas using the smallest vs largest block size present (not
    hard-coded to 16 and 32).
    """
    summary = summary.copy()
    if "last_block_slack_tokens" not in summary.columns:
        T = (
            (summary["total_prompt_tokens"] + summary["total_output_tokens"])
            / summary["num_sequences"]
        ).astype(int)
        B = summary["block_size"].astype(int)
        summary["last_block_slack_tokens"] = (-T % B)

    blocks = sorted(summary["block_size"].astype(int).unique())
    if len(blocks) < 2:
        warnings.warn("Only one block_size in data — delta columns omitted.", stacklevel=2)

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

    def pct_delta(new: float, old: float) -> float:
        return 100.0 * (new - old) / old if old else float("nan")

    rows_out: list[dict] = []
    b_small, b_large = blocks[0], blocks[-1]
    for L in pivot_tp.index:
        row: dict = {"prompt_len_cfg": int(L)}
        for b in blocks:
            if b in pivot_tp.columns and pd.notna(pivot_tp.loc[L, b]):
                row[f"throughput_b{int(b)}"] = float(pivot_tp.loc[L, b])
            if b in pivot_peak.columns and pd.notna(pivot_peak.loc[L, b]):
                row[f"peak_mem_mb_b{int(b)}"] = float(pivot_peak.loc[L, b])
            if b in pivot_slack.columns and pd.notna(pivot_slack.loc[L, b]):
                row[f"last_block_slack_b{int(b)}"] = float(pivot_slack.loc[L, b])
        if len(blocks) >= 2:
            if b_small in pivot_tp.columns and b_large in pivot_tp.columns:
                row[f"tp_pct_delta_b{b_small}_vs_b{b_large}"] = pct_delta(
                    float(pivot_tp.loc[L, b_small]),
                    float(pivot_tp.loc[L, b_large]),
                )
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def run_h4_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load H4 summaries, validate the grid, enrich metrics, write figures, and
    return `(summary, comparison)`.
    """
    summary = load_h4_summaries()
    validate_h4_design_grid(summary)

    T = (
        (summary["total_prompt_tokens"] + summary["total_output_tokens"])
        / summary["num_sequences"]
    ).astype(int)
    B = summary["block_size"].astype(int)
    summary = summary.copy()
    summary["avg_seq_tokens"] = T
    summary["last_block_slack_tokens"] = (-T % B)
    summary = attach_request_latency_spread(summary, RESULTS_ROOT)

    prompt_lengths = sorted(summary["prompt_len_cfg"].unique())

    n = len(prompt_lengths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, L in zip(axes, prompt_lengths):
        sub = summary[summary["prompt_len_cfg"] == L].sort_values("block_size")
        xs = sub["block_size"].astype(str).tolist()
        ys = sub["throughput_tok_per_s"].tolist()
        nbar = len(xs)
        colors = [plt.cm.viridis(i / max(1, nbar - 1)) for i in range(nbar)]
        bars = ax.bar(xs, ys, color=colors, width=0.55)
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
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()

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

    peak_vals = summary["peak_gpu_mem_used_mb"].dropna().unique()
    if len(peak_vals) == 1:
        fig.text(
            0.5,
            0.02,
            f"Note: peak_gpu_mem_used_mb is constant ({peak_vals[0]:.1f} MB) "
            "across all loaded runs.",
            ha="center",
            fontsize=8,
            color="0.35",
        )

    plt.suptitle("H4: Reported GPU memory vs block size", y=1.02)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    out_mem = RESULTS_ROOT / "h4_memory_by_block.png"
    plt.savefig(out_mem, dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L in prompt_lengths:
        sub = summary[summary["prompt_len_cfg"] == L].sort_values("block_size")
        ax.plot(
            sub["block_size"],
            sub["last_block_slack_tokens"],
            marker="s",
            label=f"L = {L}",
        )
    ax.set_xlabel("KV block size")
    ax.set_ylabel("Last-block slack (token slots, model summary)")
    ax.set_title("H4: Logical last-block slack vs block size (see module caveats)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "h4_last_block_slack.png", dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()

    comparison = build_h4_comparison_table(summary)
    return summary, comparison


# %%
if __name__ == "__main__":
    summary, comparison = run_h4_analysis()
