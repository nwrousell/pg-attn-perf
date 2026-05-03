#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hypothesis_dir> [--nsys]"
    exit 1
fi

USE_NSYS=false
if [ "${2:-}" = "--nsys" ]; then
    USE_NSYS=true
fi

for config in configs/$1/*.yaml; do
    echo "=== Running $config ==="
    if $USE_NSYS; then
        source .venv/bin/activate
        ./run_nsys.sh "$config"
    else
        uv run runner.py "$config"
    fi
done
