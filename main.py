from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="facebook/opt-125m")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

    prompts = ["The future of AI is"]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated!r}")


if __name__ == "__main__":
    main()
