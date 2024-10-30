from vllm import LLM, SamplingParams
import time
from typing import List
from util import estimate_shared_prefix_tokens, shared_prefix


def generate_outputs(llm, prompts: List[str], sampling_params):
    formatted_prompts = [f"<s>[INST] {prompt} [/INST]" for prompt in prompts]
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def main():
    # Common LLM configuration
    llm_config = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "tensor_parallel_size": 1,
        "max_model_len": 1024,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 2048,
        "use_v2_block_manager": True,
        "gpu_memory_utilization": 0.7,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
    }

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=8,
        stop=["</s>", "[/INST]"],
    )

    topics = [
        "quantum computing",
        "neural networks",
        "blockchain",
        "AI ethics",
        "cloud systems",
        "cybersecurity",
        "data science",
        "edge computing",
        "deep learning",
        "robotics",
    ] * 10

    prompts = [f"{shared_prefix} {topic}." for topic in topics]

    # Estimate tokens in shared prefix
    token_count = estimate_shared_prefix_tokens(llm_config["model"], shared_prefix)
    print(f"\nEstimated tokens in shared prefix: {token_count}")

    # Test shared prefix approach
    print("\nInitializing model with prefix sharing...")
    llm = LLM(**llm_config, enable_prefix_caching=True)

    # Warmup run
    print("Performing warmup run...")
    _ = generate_outputs(llm, prompts[:2], sampling_params)

    print("Running performance test...")
    start_time = time.time()
    shared_results = generate_outputs(llm, prompts, sampling_params)
    shared_time = time.time() - start_time
    del llm

    # Test regular approach (prefix sharing disabled)
    print("\nInitializing model without prefix sharing...")
    llm_regular = LLM(**llm_config, enable_prefix_caching=False)

    # Warmup run
    print("Performing warmup run...")
    _ = generate_outputs(llm_regular, prompts[:2], sampling_params)

    print("Running performance test...")
    start_time = time.time()
    regular_results = generate_outputs(llm_regular, prompts, sampling_params)
    regular_time = time.time() - start_time
    del llm_regular

    # Print timing comparison
    print("\n=== Performance Comparison ===")
    print(f"Shared prefix execution time: {shared_time:.2f} seconds")
    print(f"Regular execution time: {regular_time:.2f} seconds")
    print(f"Time saved with prefix sharing: {regular_time - shared_time:.2f} seconds")
    print(f"Performance improvement: {(regular_time/shared_time - 1)*100:.1f}%")

    # Print sample results (first few only)
    print("\n=== Sample Generated Results ===")
    print("\nShared Prefix Results (first 2):")
    for i, (prompt, result) in enumerate(zip(prompts[:2], shared_results[:2]), 1):
        print(f"\nPrompt {i}: {prompt}")
        print(f"Response {i}: {result}")

    print("\nRegular Results (first 2):")
    for i, (prompt, result) in enumerate(zip(prompts[:2], regular_results[:2]), 1):
        print(f"\nPrompt {i}: {prompt}")
        print(f"Response {i}: {result}")


if __name__ == "__main__":
    main()
