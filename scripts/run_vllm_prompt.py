#!/usr/bin/env python3
"""
Run prompts through a local model using vLLM.

Exposes:
  load_model(model_path, tensor_parallel_size, gpu_memory_utilization) -> LLM
  run_prompt(llm, prompt, ...)                                          -> dict

vLLM uses PagedAttention for dynamic KV-cache allocation, making it
significantly faster than llama.cpp for batch inference. Models are loaded
directly from Hugging Face — no GGUF conversion needed.

Requirements:
  pip install -r requirements-vllm.txt

Platform: Linux + NVIDIA CUDA only. Not supported on macOS or AMD.

Thinking mode (--enable-thinking) is supported for models that expose
enable_thinking in their chat template (e.g. Gemma 4). The thinking block
is stripped from `response` and stored in `thinking`, matching the schema
used by run_mlx_prompt.py, run_llama_prompt.py, and frontier thinking models.

CLI usage (loads model fresh each call — use run_vllm_jsonl.py for batches):
    python run_vllm_prompt.py --model google/gemma-4-e4b-it --prompt "text"
    python run_vllm_prompt.py --model google/gemma-4-e4b-it --prompt "text" --enable-thinking
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from normalize_answer import extract_final_answer

# Gemma 4 thinking delimiters — identical to run_mlx_prompt.py and run_llama_prompt.py
_THINKING_DELIMITER = "<channel|>"
_THINKING_PREFIX = "<|channel>thought\n"


def load_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    verbose: bool = False,
):
    """
    Load a model with vLLM.

    tensor_parallel_size: number of GPUs to use (default: 1).
    gpu_memory_utilization: fraction of GPU memory to use for the model
        and KV cache (default: 0.90). Lower if you run out of VRAM.
    max_model_len: maximum sequence length (prompt + response). vLLM
        allocates KV cache dynamically up to this limit.
    """
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vllm is required: pip install -r requirements-vllm.txt\n"
            "vLLM is supported on Linux with NVIDIA CUDA only."
        )
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        disable_log_stats=not verbose,
    )


def run_prompt(
    llm,
    prompt: str,
    model_path: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> dict:
    """
    Run a single prompt and return a structured result dict compatible with
    the JSONL output schema used by run_eval_jsonl.py and score_eval.py.

    use_chat_template=True wraps the prompt in the model's instruct format.
    enable_thinking=True passes enable_thinking=True to the chat template for
    models that support it (e.g. Gemma 4). The thinking block is stripped from
    `response` via extract_final_answer() and stored separately in `thinking`.
    """
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vllm is required: pip install -r requirements-vllm.txt")

    t0 = time.time()

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if use_chat_template:
        tokenizer = llm.get_tokenizer()
        chat_kwargs = {}
        if enable_thinking:
            chat_kwargs["enable_thinking"] = True
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )
        except Exception:
            # Fall back without enable_thinking if template doesn't support it
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        outputs = llm.generate([formatted], sampling_params, use_tqdm=False)
    else:
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

    raw_response = (outputs[0].outputs[0].text or "").strip()
    elapsed = round(time.time() - t0, 3)

    response = extract_final_answer(raw_response)
    thinking = (
        raw_response.split(_THINKING_DELIMITER, 1)[0]
        .removeprefix(_THINKING_PREFIX).strip()
        if _THINKING_DELIMITER in raw_response else None
    )

    result = {
        "model": model_path,
        "prompt": prompt,
        "response": response,
        "generation_kwargs": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_chat_template": use_chat_template,
            "enable_thinking": enable_thinking,
        },
        "elapsed_s": elapsed,
    }
    if thinking is not None:
        result["thinking"] = thinking
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a prompt through a local model using vLLM."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path (e.g. google/gemma-4-e4b-it)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default: 0.90)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum sequence length in tokens (default: 8192)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template — use raw prompt (base models)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable chain-of-thought thinking for supported models (e.g. Gemma 4)")
    args = parser.parse_args()

    llm = load_model(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        verbose=True,
    )
    result = run_prompt(
        llm,
        prompt=args.prompt,
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_chat_template=not args.no_chat_template,
        enable_thinking=args.enable_thinking,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
