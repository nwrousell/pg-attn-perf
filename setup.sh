# first grab a beefy interactive gpu node with `interact -q gpu -g 1 -n 12 -m 40g -t 4:00:00`

module load cuda/12.9
uv venv --python 3.12 --seed
source .venv/bin/activate

# these make the build run faster
export MAX_JOBS=12
export VLLM_USE_PRECOMPILED=1

uv pip install vllm --torch-backend=cu129 -v