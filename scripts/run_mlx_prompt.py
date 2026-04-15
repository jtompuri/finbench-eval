#!/usr/bin/env python3
"""
Run prompts through a local MLX model using the mlx_lm Python API.

Exposes:
  load_model(model_path)            -> (model, tokenizer)
  run_prompt(model, tokenizer, ...) -> dict

CLI usage (loads model fresh each call — use run_eval_jsonl.py for batches):
    python run_mlx_prompt.py --model /path/to/model --prompt "text" [--max-tokens 512] [--temperature 0.0]
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

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def load_model(model_path: str):
    """Load an MLX model and tokenizer from a local filesystem path."""
    return load(model_path)


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool = False) -> str:
    """Wrap a plain prompt string in the model's chat template.

    enable_thinking=False suppresses internal chain-of-thought for models that
    support the parameter (e.g. Gemma 4).  Falls back silently for models that
    don't accept the keyword argument.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Tokenizer does not support enable_thinking — ignore and proceed normally
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def run_prompt(model, tokenizer, prompt: str, model_path: str,
               max_tokens: int = 512, temperature: float = 0.0,
               use_chat_template: bool = True,
               enable_thinking: bool = False) -> dict:
    """Run a single prompt and return a structured result dict.

    enable_thinking=True activates chain-of-thought for models that support it
    (e.g. Gemma 4).  extract_final_answer() from normalize_answer.py strips the
    thinking block from `response` — the same logic used for frontier models.
    The raw thinking text is stored separately in `thinking`.
    """
    if use_chat_template:
        formatted_prompt = apply_chat_template(tokenizer, prompt,
                                               enable_thinking=enable_thinking)
    else:
        formatted_prompt = prompt

    t0 = time.time()
    sampler = make_sampler(temp=temperature)
    raw_response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        verbose=False,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    elapsed = round(time.time() - t0, 3)

    # Reuse the same extract_final_answer() used by frontier models and score_eval.py
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
    parser = argparse.ArgumentParser(description="Run a prompt through a local MLX model.")
    parser.add_argument("--model", required=True, help="Path to local MLX model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template wrapping (use raw prompt)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    result = run_prompt(
        model, tokenizer,
        prompt=args.prompt,
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_chat_template=not args.no_chat_template,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
