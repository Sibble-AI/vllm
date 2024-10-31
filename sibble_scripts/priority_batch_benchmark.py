from vllm import LLM, SamplingParams
import time
from typing import List, Dict
import heapq
from dataclasses import dataclass
from queue import PriorityQueue
import numpy as np


@dataclass(order=True)
class PrioritizedRequest:
    priority: float
    prompt: str = None


def create_tree_search_prompts(depth: int = 3, branches: int = 3) -> Dict[str, float]:
    """Creates a set of prompts simulating tree search with assigned priorities"""
    prompts_with_priority = {}
    base_prompt = "Analyze the following solution path:"

    # Create one "golden" high-priority path
    golden_priority = 0.95  # Very high priority solution
    golden_path = "Sustainable vertical farming + Distribution optimization + Community engagement"
    prompts_with_priority[f"{base_prompt} Solution: {golden_path}"] = golden_priority

    # Create other paths with lower priorities
    solutions = [
        "Local food production networks",
        "Technology-enabled distribution",
        "Policy and infrastructure changes",
        "Education and training programs",
    ]

    for level in range(depth):
        for branch in range(branches):
            # Lower levels have potentially higher priorities (simulating getting closer to solution)
            base_priority = (level + 1) / depth  # Increases with depth
            priority = (
                np.random.beta(2, 5) * base_priority
            )  # More low priorities than high
            solution = np.random.choice(solutions)
            prompt = f"{base_prompt} Level {level}, Branch {branch}: {solution}"
            prompts_with_priority[prompt] = priority

    return prompts_with_priority


def batch_requests_priority(
    prompts_dict: Dict[str, float], batch_size: int
) -> List[List[str]]:
    """Groups requests into batches, prioritizing higher scores"""
    # Pre-sort all items at once instead of using a priority queue
    sorted_prompts = sorted(prompts_dict.items(), key=lambda x: x[1], reverse=True)
    prompts = [prompt for prompt, _ in sorted_prompts]

    # Use the same list comprehension as FIFO for batching
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


def batch_requests_fifo(
    prompts_dict: Dict[str, float], batch_size: int
) -> List[List[str]]:
    """Groups requests into batches in FIFO order"""
    prompts = list(prompts_dict.keys())
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


def generate_outputs(llm, prompts: List[str], sampling_params):
    formatted_prompts = [f"<s>[INST] {prompt} [/INST]" for prompt in prompts]
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def main():
    llm_config = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "tensor_parallel_size": 1,
        "max_model_len": 1024,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.7,
    }

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=64,
        stop=["</s>", "[/INST]"],
    )

    # Create tree search simulation prompts
    prompts_with_priority = create_tree_search_prompts(depth=3, branches=3)
    batch_size = 4

    # Initialize model
    llm = LLM(**llm_config)

    # Test priority-based batching with early stopping
    print("\n=== Priority-based Search Path ===")
    priority_batches = batch_requests_priority(prompts_with_priority, batch_size)

    priority_results = []
    start_time = time.time()
    found_solution = False

    for batch_idx, batch in enumerate(priority_batches):
        results = generate_outputs(llm, batch, sampling_params)
        priority_results.extend(results)

        # Print batch information and results
        print(f"\nBatch {batch_idx + 1}:")
        for prompt, result in zip(batch, results):
            priority = prompts_with_priority[prompt]
            print(f"\nPriority: {priority:.3f}")
            print(f"Prompt: {prompt}")
            print(f"Response: {result[:100]}...")  # Truncate long responses

            if priority > 0.9:  # Found high-priority solution
                print("\n🎯 Found high-priority solution!")
                found_solution = True
                break
        if found_solution:
            break

    priority_time = time.time() - start_time

    # Test FIFO batching
    print("\n=== FIFO Search Path ===")
    fifo_batches = batch_requests_fifo(prompts_with_priority, batch_size)

    fifo_results = []
    start_time = time.time()
    for batch_idx, batch in enumerate(fifo_batches):
        results = generate_outputs(llm, batch, sampling_params)
        fifo_results.extend(results)

        # Print batch information and results
        print(f"\nBatch {batch_idx + 1}:")
        for prompt, result in zip(batch, results):
            print(f"\nPriority: {prompts_with_priority[prompt]:.3f}")
            print(f"Prompt: {prompt}")
            print(f"Response: {result[:100]}...")

    fifo_time = time.time() - start_time

    # Print final comparison
    print("\n=== Performance Comparison ===")
    print(
        f"Priority-based batching time (with early stopping): {priority_time:.2f} seconds"
    )
    print(f"FIFO batching time: {fifo_time:.2f} seconds")
    print(f"Priority batches processed: {len(priority_results)} outputs")
    print(f"FIFO batches processed: {len(fifo_results)} outputs")


if __name__ == "__main__":
    main()
