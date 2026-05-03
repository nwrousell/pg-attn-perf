#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

CONFIG="$1"
NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['name'])")
OUTDIR="results/$NAME"
mkdir -p "$OUTDIR"

echo "=== nsys profiling: $NAME ==="

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
nsys profile \
    -o "$OUTDIR/nsys_report" \
    --trace-fork-before-exec=true \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --force-overwrite=true \
    uv run runner.py "$CONFIG"

echo "=== Exporting kernel summary to CSV ==="

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --force-export=true \
    --force-overwrite=true \
    -o "$OUTDIR/nsys_kern" \
    "$OUTDIR/nsys_report.nsys-rep"

echo "Done. Kernel summary -> $OUTDIR/nsys_kern_cuda_gpu_kern_sum.csv"