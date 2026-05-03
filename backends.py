from abc import ABC, abstractmethod
import time

import pandas as pd
import torch


class Backend(ABC):
    @abstractmethod
    def setup(self, config: dict):
        ...

    @abstractmethod
    def run(self, workload: list[dict]) -> pd.DataFrame:
        """Run workload, return per-request stats DataFrame."""
        ...

    @abstractmethod
    def get_tokenizer(self):
        ...


class VLLMBackend(Backend):
    def setup(self, config: dict):
        from vllm import LLM

        self.dtype = config.get("dtype", "float16")
        self.block_size = config.get("block_size", 16)
        self.llm = LLM(
            model=config["model"],
            dtype=self.dtype,
            block_size=self.block_size,
            attention_config={"backend": "FLASH_ATTN"},
        )
        self._tokenizer = self.llm.get_tokenizer()

    def get_tokenizer(self):
        return self._tokenizer

    def run(self, workload: list[dict]) -> pd.DataFrame:
        from vllm import SamplingParams

        prompts = [w["prompt"] for w in workload]
        # Each request can have its own max_tokens via per-request SamplingParams
        sampling_params = [
            SamplingParams(max_tokens=w["max_tokens"], ignore_eos=True)
            for w in workload
        ]

        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params)
        total_time = time.perf_counter() - start

        rows = []
        for i, output in enumerate(outputs):
            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            rows.append({
                "seq_id": i,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "latency_s": total_time,  # vLLM batches internally
                "tokens_per_sec": output_tokens / total_time,
            })

        return pd.DataFrame(rows)


class HFBackend(Backend):
    def setup(self, config: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = config["model"]
        self.dtype = getattr(torch, config.get("dtype", "float16"))

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation="sdpa", # theoretically this still uses Flash Attn 2 for prefill phase (which matches vllm prefill for a fair comparison)
            device_map="cuda",
        )
        self.model.eval()

    def get_tokenizer(self):
        return self._tokenizer

    def _estimate_max_batch_size(self) -> int:
        """Estimate max batch size that fits in GPU memory with StaticCache."""
        cfg = self.model.config
        max_len = cfg.max_position_embeddings # max seq len
        n_layers = cfg.num_hidden_layers
        n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        elem_size = 2 if self.dtype == torch.float16 else 4

        # 2 tensors (key + value) per layer, each (batch, heads, seq, head_dim)
        per_seq_bytes = 2 * n_layers * n_kv_heads * max_len * head_dim * elem_size

        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0]
        return max(1, int(free_mem * 0.8 / per_seq_bytes))

    def run(self, workload: list[dict]) -> pd.DataFrame:
        max_bs = self._estimate_max_batch_size()
        print(f"  HF max batch size for GPU memory: {max_bs}")

        rows = []
        seq_id = 0
        total_start = time.perf_counter()

        for chunk_start in range(0, len(workload), max_bs):
            chunk = workload[chunk_start:chunk_start + max_bs]
            prompts = [w["prompt"] for w in chunk]
            max_new_tokens = max(w["max_tokens"] for w in chunk)

            inputs = self._tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
            ).to("cuda")

            bs = inputs["input_ids"].shape[0]

            batch_start = time.perf_counter()
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    cache_implementation="static",
                )
            batch_time = time.perf_counter() - batch_start

            for i in range(bs):
                prompt_tokens = (inputs["attention_mask"][i] == 1).sum().item()
                generated_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
                rows.append({
                    "seq_id": seq_id,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": generated_tokens,
                    "latency_s": batch_time,
                    "tokens_per_sec": generated_tokens / batch_time,
                })
                seq_id += 1

            del inputs, output_ids
            torch.cuda.empty_cache()

        total_time = time.perf_counter() - total_start
        num_batches = (len(workload) + max_bs - 1) // max_bs
        print(f"  Processed {len(workload)} sequences in {num_batches} batches")

        return pd.DataFrame(rows)


def create_backend(config: dict) -> Backend:
    backend_type = config["backend"]
    if backend_type == "vllm":
        backend = VLLMBackend()
    elif backend_type == "hf":
        backend = HFBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
    backend.setup(config)
    return backend
