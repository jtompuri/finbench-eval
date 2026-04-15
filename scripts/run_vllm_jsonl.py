#!/usr/bin/env python3
"""
Load a JSONL dataset, run each prompt through a local model via vLLM,
and save outputs as JSONL.

The model is loaded once and reused for all prompts. Output schema is
identical to run_eval_jsonl.py so that score_eval.py, aggregate_results.py,
and export_hf_dataset.py work without modification.

vLLM uses PagedAttention for dynamic KV-cache allocation — no --n-ctx
pre-allocation needed. Models are loaded directly from Hugging Face.

Platform: Linux + NVIDIA CUDA only. Not supported on macOS or AMD.

Usage — full dataset:
    python scripts/run_vllm_jsonl.py \\
        --model google/gemma-4-e4b-it \\
        --input data/finbench_combined_v1.jsonl \\
        --output outputs/combined_gemma4e4b_vllm.jsonl

Usage — small test run (5 items, verbose):
    python scripts/run_vllm_jsonl.py \\
        --model google/gemma-4-e4b-it \\
        --input data/finbench_combined_v1.jsonl \\
        --output outputs/test_gemma4e4b_vllm.jsonl \\
        --n 5 --verbose

Usage — resume interrupted run:
    python scripts/run_vllm_jsonl.py \\
        --model google/gemma-4-e4b-it \\
        --input data/finbench_combined_v1.jsonl \\
        --output outputs/combined_gemma4e4b_vllm.jsonl \\
        --resume
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_vllm_prompt import load_model, run_prompt, run_batch
from runner_utils import (
    LiveStats,
    SUMMARY_EVERY,
    build_resume_set,
    load_jsonl,
    load_run_settings,
    resolve_max_tokens,
    resolve_temperature,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a JSONL dataset through a local model using vLLM."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path (e.g. google/gemma-4-e4b-it)")
    parser.add_argument("--model-key", default=None,
                        help="Model key from configs/run_settings.yaml for max_tokens lookup. "
                             "Optional — use --max-tokens to override directly.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens (default: resolved from configs/run_settings.yaml)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from configs/run_settings.yaml)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default: 0.90)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum sequence length in tokens (default: 8192)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template — use raw prompt (base models only)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable chain-of-thought thinking for supported models (e.g. Gemma 4). "
                             "Thinking tokens are stored in a separate 'thinking' field; "
                             "the final answer is extracted into 'response'.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing code from the model repository (use with caution)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N items (useful for smoke tests)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-item response and running MCF accuracy")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already present in the output file")
    parser.add_argument("--batch", action="store_true",
                        help="Submit all prompts in a single vLLM generate() call for maximum "
                             "GPU throughput via continuous batching. Results are written after "
                             "the full batch completes — intermediate saving is not available.")
    args = parser.parse_args()

    settings    = load_run_settings()
    max_tokens  = resolve_max_tokens(settings, args.model_key, args.max_tokens)
    temperature = resolve_temperature(settings, args.temperature)

    items = load_jsonl(args.input)
    if args.n is not None:
        items = items[:args.n]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    if args.resume:
        completed_ids = build_resume_set(output_path)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} items already completed, skipping.")

    pending = [it for it in items if it.get("id") not in completed_ids]
    if len(pending) < len(items):
        print(f"Items to run: {len(pending)} / {len(items)}")

    if not pending:
        print("All items already completed — nothing to do.")
        return

    print(f"Loading model: {args.model}")
    print(f"max_tokens={max_tokens}  temperature={temperature}  "
          f"max_model_len={args.max_model_len}  "
          f"gpu_memory_utilization={args.gpu_memory_utilization}")

    llm = load_model(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    use_chat_template = not args.no_chat_template
    enable_thinking   = args.enable_thinking
    run_meta = {
        "model":                  args.model,
        "model_key":              args.model_key,
        "backend":                "vllm",
        "input":                  args.input,
        "max_tokens":             max_tokens,
        "temperature":            temperature,
        "tensor_parallel_size":   args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len":          args.max_model_len,
        "use_chat_template":      use_chat_template,
        "enable_thinking":        enable_thinking,
        "batch_mode":             args.batch,
        "started_at":             datetime.now(timezone.utc).isoformat(),
    }

    stats     = LiveStats() if args.verbose else None
    open_mode = "a" if completed_ids else "w"
    n_done    = len(completed_ids)
    n_total   = len(completed_ids) + len(pending)

    with open(output_path, open_mode, encoding="utf-8") as out_f:

        if args.batch:
            # --- Batch mode: submit all prompts at once for continuous batching ---
            print(f"Batch mode: submitting {len(pending)} prompts to vLLM in one call...")
            batch_results = run_batch(
                llm,
                prompts=[it.get("prompt", "") for it in pending],
                model_path=args.model,
                max_tokens=max_tokens,
                temperature=temperature,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            pbar = tqdm(zip(pending, batch_results), total=len(pending),
                        desc=Path(args.model).name)
            for item, result in pbar:
                record = {**item, **result, "run_meta": run_meta}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_done += 1
                if stats is not None:
                    result_str  = stats.update(item, result.get("response", ""))
                    elapsed     = result.get("elapsed_s")
                    elapsed_str = f"  {elapsed:.1f}s" if elapsed else ""
                    tqdm.write(
                        f"[{n_done:>4}/{n_total}] {item.get('id', '?'):<40} "
                        f"{result_str}{elapsed_str}"
                    )

        else:
            # --- Sequential mode: one prompt at a time, flush after each ---
            pbar = tqdm(pending, desc=Path(args.model).name)
            for item in pbar:
                result = run_prompt(
                    llm,
                    prompt=item.get("prompt", ""),
                    model_path=args.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_chat_template=use_chat_template,
                    enable_thinking=enable_thinking,
                )
                record = {**item, **result, "run_meta": run_meta}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1

                if stats is not None:
                    result_str  = stats.update(item, result.get("response", ""))
                    elapsed     = result.get("elapsed_s")
                    elapsed_str = f"  {elapsed:.1f}s" if elapsed else ""
                    tqdm.write(
                        f"[{n_done:>4}/{n_total}] {item.get('id', '?'):<40} "
                        f"{result_str}{elapsed_str}"
                    )
                    if n_done % SUMMARY_EVERY == 0:
                        for line in stats.summary_lines():
                            tqdm.write(line)

    print(f"Saved {n_done} results to {output_path} ({len(pending)} new)")

    if stats is not None:
        print()
        for line in stats.summary_lines():
            print(line)


if __name__ == "__main__":
    main()
