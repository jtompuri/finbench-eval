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

Thinking mode (--enable-thinking) is supported for models that have an
enable_thinking parameter in their Jinja2 chat template (e.g. Gemma 4).
The thinking block is stripped from `response` and stored in `thinking`,
matching the schema used by run_mlx_prompt.py and frontier thinking models.

Installation:
  # CPU-only (all platforms)
  pip install llama-cpp-python

  # NVIDIA GPU (Linux / Windows)
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

  # Apple Silicon (Metal)
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall

CLI usage (loads model fresh each call — use run_llama_jsonl.py for batches):
    python run_llama_prompt.py --model /path/to/model.gguf --prompt "text"
    python run_llama_prompt.py --model /path/to/model.gguf --prompt "text" --enable-thinking
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from normalize_answer import extract_final_answer
from runner_utils import THINKING_DELIMITER as _THINKING_DELIMITER, \
                         THINKING_PREFIX    as _THINKING_PREFIX

# Special tokens that mark the end of a model turn.  llama-cpp-python normally
# stops generation before these, but we strip them defensively from raw llm()
# output to match the clean content returned by create_chat_completion().
_STOP_TOKENS = ("<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>")


def _strip_stop_tokens(text: str) -> str:
    """Strip trailing model stop tokens from raw llm() completion output."""
    text = text.strip()
    for tok in _STOP_TOKENS:
        if text.endswith(tok):
            text = text[: -len(tok)].rstrip()
    return text


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


def apply_chat_template(llm, prompt: str, enable_thinking: bool = False) -> str:
    """Apply the model's Jinja2 chat template from GGUF metadata.

    When enable_thinking=True, passes that variable to the template so models
    that support it (e.g. Gemma 4) activate chain-of-thought generation.
    Falls back gracefully for models whose templates do not accept the variable.
    """
    template_str = llm.metadata.get("tokenizer.chat_template", "")
    if not template_str:
        # Simple fallback for models without an embedded template
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    try:
        import jinja2
    except ImportError:
        raise ImportError(
            "jinja2 is required for thinking mode: pip install jinja2\n"
            "(It is typically installed automatically with llama-cpp-python.)"
        )

    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    # Some templates call raise_exception() — provide a no-op shim
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))

    template = env.from_string(template_str)
    messages = [{"role": "user", "content": prompt}]

    # Try with enable_thinking first; fall back without if the template rejects it
    for kwargs in (
        {"enable_thinking": enable_thinking},
        {},
    ):
        try:
            rendered = template.render(
                messages=messages,
                add_generation_prompt=True,
                bos_token="<bos>",
                eos_token="<eos>",
                **kwargs,
            )
            # llama-cpp-python adds its own BOS token when using the raw llm()
            # completion endpoint, so strip any leading <bos> from the rendered
            # template to avoid a duplicate BOS warning.
            if rendered.startswith("<bos>"):
                rendered = rendered[len("<bos>"):]
            return rendered
        except jinja2.exceptions.UndefinedError:
            continue

    # Last-resort fallback
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


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

    use_chat_template=True wraps the prompt in the model's instruct format via
    apply_chat_template(), passing enable_thinking explicitly to the Jinja2
    template in both thinking and non-thinking mode.  This keeps the prompt
    format consistent with the MLX path (run_mlx_prompt.py) and avoids the
    asymmetry of using create_chat_completion() for non-thinking runs.

    enable_thinking=True activates chain-of-thought for models that support it
    (e.g. Gemma 4). The thinking block is stripped from `response` via
    extract_final_answer() and stored separately in `thinking`, matching the
    schema produced by run_mlx_prompt.py and frontier thinking models.
    """
    t0 = time.time()

    # Always use apply_chat_template + raw llm() completion so that
    # enable_thinking is passed to the Jinja2 template explicitly in both
    # thinking and non-thinking mode.  This keeps the prompt format identical
    # between the two paths and avoids relying on create_chat_completion's
    # internal template rendering which does not expose template variables.
    if use_chat_template:
        formatted = apply_chat_template(llm, prompt, enable_thinking=enable_thinking)
    else:
        formatted = prompt
    output = llm(formatted, max_tokens=max_tokens, temperature=temperature)
    raw_response = _strip_stop_tokens((output["choices"][0]["text"] or "").strip())

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
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable chain-of-thought thinking for supported models (e.g. Gemma 4)")
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
        enable_thinking=args.enable_thinking,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
