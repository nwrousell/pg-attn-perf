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
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    python runner.py "$CONFIG"

echo "=== Exporting nsys stats to CSV ==="

nsys stats \
    --format csv \
    -o "$OUTDIR/nsys_stats" \
    "$OUTDIR/nsys_report.nsys-rep"

echo "Done. Results in $OUTDIR/"
