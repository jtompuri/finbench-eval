#!/usr/bin/env python3
"""
Run prompts through a local GGUF model using llama-cpp-python.

Exposes:
  load_model(model_path, n_gpu_layers, n_ctx) -> Llama
  run_prompt(llm, prompt, ...)               -> dict

Compatible with NVIDIA (CUDA), AMD (ROCm/Vulkan), Apple Silicon (Metal),
and CPU-only inference. GPU offloading is controlled by --n-gpu-layers:
  -1  offload all layers to GPU (default, fastest)
   0  CPU-only (no GPU required)
   N  offload N layers (partial GPU, useful when VRAM is limited)

Installation:
  # CPU-only (all platforms)
  pip install llama-cpp-python

  # NVIDIA GPU (Linux / Windows)
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

  # Apple Silicon (Metal)
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall

CLI usage (loads model fresh each call — use run_llama_jsonl.py for batches):
    python run_llama_prompt.py --model /path/to/model.gguf --prompt "text"
"""
import argparse
import json
import time
from pathlib import Path


def load_model(
    model_path: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    verbose: bool = False,
):
    """
    Load a GGUF model with llama-cpp-python.

    n_gpu_layers=-1 offloads all layers to GPU (recommended when GPU is available).
    n_gpu_layers=0  runs on CPU only.
    n_ctx sets the context window size in tokens.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is required: pip install llama-cpp-python\n"
            "For GPU support see: https://github.com/abetlen/llama-cpp-python"
        )
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
    )


def run_prompt(
    llm,
    prompt: str,
    model_path: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    use_chat_template: bool = True,
) -> dict:
    """
    Run a single prompt and return a structured result dict compatible with
    the JSONL output schema used by run_eval_jsonl.py and score_eval.py.

    use_chat_template=True wraps the prompt in the model's instruct format
    via llama-cpp-python's built-in chat completion endpoint. Set to False
    for base (non-instruct) models or when the prompt is already formatted.
    """
    t0 = time.time()

    if use_chat_template:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = (output["choices"][0]["message"].get("content") or "").strip()
    else:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = (output["choices"][0]["text"] or "").strip()

    elapsed = round(time.time() - t0, 3)

    return {
        "model": model_path,
        "prompt": prompt,
        "response": response,
        "generation_kwargs": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_chat_template": use_chat_template,
        },
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a prompt through a local GGUF model (llama-cpp-python)."
    )
    parser.add_argument("--model", required=True,
                        help="Path to GGUF model file (e.g. gemma-3-27b-it-Q4_K_M.gguf)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers to offload: -1=all (default), 0=CPU only")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="Context window size in tokens (default: 4096)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template — use raw prompt (base models)")
    args = parser.parse_args()

    llm = load_model(args.model, n_gpu_layers=args.n_gpu_layers,
                     n_ctx=args.n_ctx, verbose=True)
    result = run_prompt(
        llm,
        prompt=args.prompt,
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_chat_template=not args.no_chat_template,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
