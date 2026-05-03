# %%
"""
H3: long-sequence decode with block size 16 vs 32 (vLLM, eager/torch path).

Research angle: for long contexts, smaller KV block sizes imply more logical
blocks per sequence; some attention paths can show extra reduction or
softmax-adjacent GPU work. This script compares end-to-end throughput, optional
per-request latency spread, and Nsight CUDA kernel summaries under results/h3_*.

Limitations (read before interpreting plots):
- Nsight "Time (%)" is the share of *captured GPU kernel time* in this export, not
  full wall time; rows are kernel-name aggregates from nsys_kern_cuda_gpu_kern_sum.csv.
- Softmax is often *fused* into Flash / other kernels; separate `cunn_SoftMaxForward`
  rows can under-count logical softmax work. We therefore report both narrow
  matches (SoftMax string, splitKreduce, flash_fwd_splitkv) and a *deduplicated
  union* bucket (softmax family ∪ split-K ∪ Flash-related names).
- Kernel names vary by PyTorch / vLLM / CUDA version; substring lists are best-effort.
- One capture per config: no statistical replication unless you add more runs.
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
# e.g. h3_vllm_b16_long
H3_BLOCK_PATTERN = re.compile(r"_b(\d+)_")

# Row matches *any* pattern → counted once toward the union bucket.
SOFTMAX_FAMILY_REGEX = (
    r"SoftMax|softmax|logsumexp|LogSumExp|masked_softmax|MaskedSoftmax"
)
FLASH_FAMILY_REGEX = r"flash::|flash_fwd|Flash_attn|flash_attn"


def _parse_block_size(dirname: str) -> int:
    m = H3_BLOCK_PATTERN.search(dirname)
    if not m:
        raise ValueError(f"Could not parse block size from directory name: {dirname}")
    return int(m.group(1))


def load_h3_summaries(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("h3_"):
            continue
        summary_path = d / "summary.csv"
        if not summary_path.is_file():
            continue
        df = pd.read_csv(summary_path)
        if len(df) != 1:
            raise ValueError(f"Expected one row in {summary_path}, got {len(df)}")
        row = df.iloc[0].to_dict()
        row["run_dir"] = d.name
        b = _parse_block_size(d.name)
        row["block_from_name"] = b
        if int(row["block_size"]) != b:
            raise ValueError(
                f"block_size mismatch: dir {d.name} vs summary {row['block_size']}"
            )
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No h3_* runs under {results_root.resolve()} (need summary.csv)."
        )
    out = pd.DataFrame(rows)
    return out.sort_values("block_size").reset_index(drop=True)


def _read_nsys_kernel_sum(nsys_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(nsys_csv)
    required = {"Name", "Time (%)", "Total Time (ns)", "Instances"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{nsys_csv}: missing columns {sorted(missing)}")
    return df


def _sum_time(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    m = mask.fillna(False)
    return float(df.loc[m, "Time (%)"].sum()), float(df.loc[m, "Total Time (ns)"].sum())


def summarize_nsys_kernels(nsys_csv: Path) -> dict[str, float]:
    """Aggregate Nsight 'CUDA GPU Kernel Summary' rows (narrow + union buckets)."""
    df = _read_nsys_kernel_sum(nsys_csv)
    name = df["Name"].astype(str)

    def sums_substr(substr: str) -> tuple[float, float]:
        m = name.str.contains(substr, case=False, regex=False)
        return _sum_time(df, m)

    softmax_pct, softmax_ns = sums_substr("SoftMax")
    splitk_pct, splitk_ns = sums_substr("splitKreduce")
    flash_splitkv_pct, flash_splitkv_ns = sums_substr("flash_fwd_splitkv")

    m_softmax_family = name.str.contains(SOFTMAX_FAMILY_REGEX, case=False, regex=True)
    m_flash_family = name.str.contains(FLASH_FAMILY_REGEX, case=False, regex=True)
    m_splitk = name.str.contains("splitKreduce", case=False, regex=False)
    m_union = m_softmax_family | m_flash_family | m_splitk
    union_pct, union_ns = _sum_time(df, m_union)

    reported_total_pct = float(df["Time (%)"].sum())

    return {
        "nsys_softmax_time_pct": softmax_pct,
        "nsys_softmax_total_ns": softmax_ns,
        "nsys_splitk_reduce_time_pct": splitk_pct,
        "nsys_splitk_reduce_total_ns": splitk_ns,
        "nsys_flash_splitkv_time_pct": flash_splitkv_pct,
        "nsys_flash_splitkv_total_ns": flash_splitkv_ns,
        "nsys_softmax_plus_splitk_time_pct": softmax_pct + splitk_pct,
        "nsys_softmax_plus_splitk_total_ns": softmax_ns + splitk_ns,
        "nsys_softmax_flash_splitk_union_time_pct": union_pct,
        "nsys_softmax_flash_splitk_union_total_ns": union_ns,
        "nsys_reported_kernel_time_pct_sum": reported_total_pct,
    }


NSYS_METRIC_KEYS: tuple[str, ...] = (
    "nsys_softmax_time_pct",
    "nsys_softmax_total_ns",
    "nsys_splitk_reduce_time_pct",
    "nsys_splitk_reduce_total_ns",
    "nsys_flash_splitkv_time_pct",
    "nsys_flash_splitkv_total_ns",
    "nsys_softmax_plus_splitk_time_pct",
    "nsys_softmax_plus_splitk_total_ns",
    "nsys_softmax_flash_splitk_union_time_pct",
    "nsys_softmax_flash_splitk_union_total_ns",
    "nsys_reported_kernel_time_pct_sum",
)


def _empty_nsys_metrics() -> dict[str, float]:
    return {k: float("nan") for k in NSYS_METRIC_KEYS}


def attach_nsys(summary: pd.DataFrame, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    extra_rows: list[dict[str, float]] = []
    for _, r in summary.iterrows():
        kern = results_root / r["run_dir"] / "nsys_kern_cuda_gpu_kern_sum.csv"
        if kern.is_file():
            extra_rows.append(summarize_nsys_kernels(kern))
        else:
            extra_rows.append(_empty_nsys_metrics())
    extras = pd.DataFrame(extra_rows, columns=list(NSYS_METRIC_KEYS))
    return pd.concat([summary.reset_index(drop=True), extras], axis=1)


def nsys_top_kernels(nsys_csv: Path, k: int = 18) -> pd.DataFrame:
    df = _read_nsys_kernel_sum(nsys_csv)
    out = df.nlargest(k, "Time (%)")[
        ["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Name"]
    ].reset_index(drop=True)
    return out


def load_requests_stats(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "requests.csv"
    if not path.is_file():
        return None
    r = pd.read_csv(path)
    if "latency_s" not in r.columns:
        return None
    lat = r["latency_s"].astype(float)
    return {
        "req_latency_mean_s": float(lat.mean()),
        "req_latency_std_s": float(lat.std(ddof=0)),
        "req_latency_p95_s": float(lat.quantile(0.95)),
        "req_n": int(len(lat)),
    }


def attach_requests(summary: pd.DataFrame, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    extras: list[dict[str, float]] = []
    keys: list[str] | None = None
    for _, r in summary.iterrows():
        st = load_requests_stats(results_root / r["run_dir"])
        if st is None:
            st = {
                "req_latency_mean_s": float("nan"),
                "req_latency_std_s": float("nan"),
                "req_latency_p95_s": float("nan"),
                "req_n": 0,
            }
        if keys is None:
            keys = list(st.keys())
        extras.append(st)
    return pd.concat([summary.reset_index(drop=True), pd.DataFrame(extras)], axis=1)


def run_h3_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load H3 results, print kernel tables, write figures, return `(full, comparison)`."""
    summary = load_h3_summaries()
    full = attach_nsys(attach_requests(summary))

    for _, row in full.iterrows():
        s = row.get("nsys_reported_kernel_time_pct_sum", float("nan"))
        if pd.notna(s) and not (98.0 <= float(s) <= 101.0):
            warnings.warn(
                f"{row['run_dir']}: Nsight Time (%) column sums to {s:.2f} "
                f"(expected ~100 for a full kernel summary)."
            )

    for _, row in full.iterrows():
        p = RESULTS_ROOT / row["run_dir"] / "nsys_kern_cuda_gpu_kern_sum.csv"
        if not p.is_file():
            continue
        print("\n=== Top GPU kernels:", row["run_dir"], "===")
        print(nsys_top_kernels(p, k=15).to_string(index=False))
        df = _read_nsys_kernel_sum(p)
        n = df["Name"].astype(str)
        sm = df[n.str.contains(SOFTMAX_FAMILY_REGEX, case=False, regex=True)]
        if len(sm):
            print("--- Rows matching softmax-family regex (may be outside top-15) ---")
            print(sm[["Time (%)", "Total Time (ns)", "Instances", "Name"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 5))
    sub = full.sort_values("block_size")
    labels = [f"block {int(b)}" for b in sub["block_size"]]
    ys = sub["throughput_tok_per_s"].tolist()
    bars = ax.bar(labels, ys, color=["tab:cyan", "tab:olive"][: len(ys)], width=0.5)
    for bar, val in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("H3: Long-sequence throughput vs KV block size")
    ax.grid(True, alpha=0.3, axis="y")
    fig.text(
        0.5,
        0.01,
        "Single scheduled batch per config; differences can be within run-to-run noise.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(RESULTS_ROOT / "h3_throughput_by_block.png", dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()

    kern_cols = [
        ("nsys_softmax_time_pct", "SoftMax (narrow)"),
        ("nsys_splitk_reduce_time_pct", "cuBLAS split-K reduce"),
        ("nsys_flash_splitkv_time_pct", "flash_fwd_splitkv (narrow)"),
        ("nsys_softmax_flash_splitk_union_time_pct", "Union: softmax family ∪ Flash ∪ split-K"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = range(len(kern_cols))
    w = 0.35
    blocks = sorted(full["block_size"].unique())
    for i, b in enumerate(blocks):
        row = full.loc[full["block_size"] == b].iloc[0]
        vals = [float(row[c]) for c, _ in kern_cols]
        ax.bar(
            [xi + (i - len(blocks) / 2 + 0.5) * w for xi in x],
            vals,
            width=w,
            label=f"block {int(b)}",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in kern_cols], rotation=18, ha="right")
    ax.set_ylabel("Time (% of rows in kernel summary)")
    ax.set_title("H3: Nsight kernel-time share (read caveats in module docstring)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    sums = full["nsys_reported_kernel_time_pct_sum"].dropna().unique()
    sum_note = (
        f"Reported ∑Time(%) ≈ {sums[0]:.2f}"
        if len(sums) == 1
        else f"Reported ∑Time(%) values: {', '.join(f'{s:.2f}' for s in sorted(set(sums)))}"
    )
    fig.text(
        0.5,
        0.01,
        sum_note + " (kernel table, not wall clock).",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(RESULTS_ROOT / "h3_nsys_softmax_splitk_flash.png", dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()

    cols = [
        "run_dir",
        "block_size",
        "throughput_tok_per_s",
        "total_time_s",
        "avg_latency_s",
        "req_latency_std_s",
        "nsys_softmax_time_pct",
        "nsys_splitk_reduce_time_pct",
        "nsys_flash_splitkv_time_pct",
        "nsys_softmax_flash_splitk_union_time_pct",
        "nsys_softmax_plus_splitk_time_pct",
        "nsys_reported_kernel_time_pct_sum",
    ]
    comparison = full[[c for c in cols if c in full.columns]].copy()

    blocks_list = sorted(full["block_size"].unique().tolist())
    if len(blocks_list) == 2:
        lo, hi = blocks_list[0], blocks_list[1]
        r_lo = full.loc[full["block_size"] == lo].iloc[0]
        r_hi = full.loc[full["block_size"] == hi].iloc[0]

        def pct_vs_ref(x: float, ref: float) -> float:
            return (x - ref) / ref * 100.0 if ref else float("nan")

        comparison_extra = pd.DataFrame(
            [
                {
                    "compare": f"b{int(lo)} vs b{int(hi)}",
                    "tp_pct_delta_lo_vs_hi": pct_vs_ref(
                        float(r_lo["throughput_tok_per_s"]),
                        float(r_hi["throughput_tok_per_s"]),
                    ),
                    "union_kernel_time_pct_delta_lo_vs_hi": float(
                        r_lo["nsys_softmax_flash_splitk_union_time_pct"]
                        - r_hi["nsys_softmax_flash_splitk_union_time_pct"]
                    ),
                    "narrow_softmax_total_ns_ratio_lo_over_hi": (
                        float(r_lo["nsys_softmax_total_ns"])
                        / float(r_hi["nsys_softmax_total_ns"])
                        if r_hi["nsys_softmax_total_ns"]
                        else float("nan")
                    ),
                }
            ]
        )
        print("\nPairwise (exactly two block sizes):\n", comparison_extra.to_string(index=False))
    elif len(blocks_list) > 2:
        print(
            f"\nNote: {len(blocks_list)} block sizes — pairwise summary skipped; "
            "inspect `full` directly."
        )

    return full, comparison


# %%
if __name__ == "__main__":
    full, comparison = run_h3_analysis()
