import random

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A stitch in time saves nine. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
    "In the beginning was the word. "
    "It was the best of times, it was the worst of times. "
    "Call me Ishmael. "
    "I think, therefore I am. "
    "To infinity and beyond. "
    "Elementary, my dear Watson. "
    "May the force be with you. "
    "Houston, we have a problem. "
    "One small step for man, one giant leap for mankind. "
    "I have a dream that one day this nation will rise up. "
    "We hold these truths to be self-evident. "
    "Four score and seven years ago our fathers brought forth on this continent. "
    "It is a truth universally acknowledged that a single man in possession of a good fortune. "
    "In a hole in the ground there lived a hobbit. "
    "It was a bright cold day in April, and the clocks were striking thirteen. "
)


def _sample_values(dist_cfg: dict, n: int, rng: random.Random) -> list[int]:
    dist = dist_cfg.get("distribution", "fixed")
    if dist == "fixed":
        val = dist_cfg.get("value", dist_cfg.get("min", 64))
        return [val] * n
    elif dist == "uniform":
        lo, hi = dist_cfg["min"], dist_cfg["max"]
        return [rng.randint(lo, hi) for _ in range(n)]
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _make_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt string that is exactly `target_tokens` tokens long."""
    filler_ids = tokenizer.encode(FILLER_TEXT * 20, add_special_tokens=False)
    # Repeat until we have enough tokens, then truncate
    while len(filler_ids) < target_tokens:
        filler_ids = filler_ids * 2
    token_ids = filler_ids[:target_tokens]
    return tokenizer.decode(token_ids)


def generate_workload(config: dict, tokenizer) -> list[dict]:
    """Generate a list of {"prompt": str, "max_tokens": int} dicts."""
    seed = config.get("seed", 42)
    rng = random.Random(seed)

    wl = config["workload"]
    n = wl["num_sequences"]

    prompt_lengths = _sample_values(wl["prompt_length"], n, rng)
    output_tokens = _sample_values(wl["output_tokens"], n, rng)

    workload = []
    for pl, ot in zip(prompt_lengths, output_tokens):
        prompt = _make_prompt(tokenizer, pl)
        workload.append({"prompt": prompt, "max_tokens": ot})

    return workload
