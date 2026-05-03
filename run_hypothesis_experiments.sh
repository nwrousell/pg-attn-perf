#!/bin/bash
set -euo pipefail

for config in configs/$1/*.yaml; do
    echo "=== Running $config ==="
    uv run runner.py "$config"
done
