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

nsys profile \
    -o "$OUTDIR/nsys_report" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    python runner.py "$CONFIG"

echo "=== Exporting kernel summary to CSV ==="

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    -o "$OUTDIR/nsys_kern" \
    "$OUTDIR/nsys_report.nsys-rep"

echo "Done. Kernel summary -> $OUTDIR/nsys_kern_cuda_gpu_kern_sum.csv"