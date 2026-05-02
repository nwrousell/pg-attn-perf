module load cuda/12.9
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto -v