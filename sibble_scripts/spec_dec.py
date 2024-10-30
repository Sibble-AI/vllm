import re
import torch
from vllm import LLM, SamplingParams


prompts = [
    "Clouds are formed by",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Common LLM configuration
llm_config = {
    "model": "facebook/opt-6.7b",
    "tensor_parallel_size": 1,
    "max_model_len": 128,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 128,
    "use_v2_block_manager": True,
}

# Run with speculative decoding
llm_spec = LLM(
    **llm_config,
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
)

print("\n=== Speculative Decoding ===")
outputs_spec = llm_spec.generate(prompts, sampling_params)
for output in outputs_spec:
    tokens_per_second = len(output.outputs[0].token_ids) / (
        output.metrics.finished_time - output.metrics.first_scheduled_time
    )
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

# Clean up memory
del llm_spec
torch.cuda.empty_cache()

# Run without speculative decoding
llm_normal = LLM(**llm_config)

print("\n=== Normal Inference ===")
outputs_normal = llm_normal.generate(prompts, sampling_params)
for output in outputs_normal:
    tokens_per_second = len(output.outputs[0].token_ids) / (
        output.metrics.finished_time - output.metrics.first_scheduled_time
    )
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
