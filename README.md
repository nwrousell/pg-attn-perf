# Paged Attention Performance Analysis

Benchmarking framework comparing vLLM (PagedAttention) vs HuggingFace Transformers (contiguous StaticCache) under configurable workloads.

## Setup

```
bash setup.sh
```

## Running experiments

Each experiment is defined by a YAML config in `configs/`. Run one with:

```bash
uv run runner.py configs/vllm_heterogeneous.yaml
```

Results are written to `results/<experiment_name>/`