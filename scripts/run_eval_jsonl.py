#!/usr/bin/env python3
"""
Load a JSONL dataset, run each prompt through a local MLX model, and save outputs as JSONL.

The model is loaded once and reused for all prompts in the file.
max_tokens is resolved from configs/run_settings.yaml based on --model-key,
or can be overridden directly with --max-tokens.

Usage:
    python run_eval_jsonl.py --model /path/to/model --model-key gemma4 \
        --input data/smoke_prompts.jsonl --output outputs/run_001.jsonl
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_mlx_prompt import load_model, run_prompt
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
    parser = argparse.ArgumentParser(description="Run a JSONL dataset through a local MLX model.")
    parser.add_argument("--model", required=True, help="Path to local MLX model directory")
    parser.add_argument("--model-key", default=None,
                        help="Model key from configs/models.yaml (e.g. gemma4, gemma3) "
                             "used to resolve max_tokens from run_settings.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens (default: resolved from run_settings.yaml)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from run_settings.yaml)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template wrapping (use raw prompt)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable chain-of-thought for models that support it (e.g. Gemma 4). "
                             "Thinking tokens are stored in a separate 'thinking' field; "
                             "'response' always contains only the final answer.")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N items after optional --subset filter "
                             "(useful for smoke tests, e.g. --n 5)")
    parser.add_argument("--subset", default=None,
                        help="Run only items whose 'task' field matches this value "
                             "(e.g. --subset arc_challenge_fi)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-item response and running MCF accuracy")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already present in the output file (resume interrupted run)")
    args = parser.parse_args()

    settings    = load_run_settings()
    max_tokens  = resolve_max_tokens(settings, args.model_key, args.max_tokens)
    temperature = resolve_temperature(settings, args.temperature)

    items = load_jsonl(args.input)
    if args.subset:
        items = [it for it in items if it.get("task") == args.subset]
        if not items:
            print(f"ERROR: no items found for --subset '{args.subset}'", file=sys.stderr)
            sys.exit(1)
        print(f"Subset '{args.subset}': {len(items)} items")
    if args.n is not None:
        items = items[:args.n]
        print(f"Limiting to first {len(items)} items (--n {args.n})")

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

    print(f"Loading model from {args.model} ...")
    print(f"max_tokens={max_tokens}  temperature={temperature}")
    model, tokenizer = load_model(args.model)

    use_chat_template = not args.no_chat_template
    enable_thinking   = args.enable_thinking
    run_meta = {
        "model":            args.model,
        "model_key":        args.model_key,
        "backend":          "mlx",
        "input":            args.input,
        "max_tokens":       max_tokens,
        "temperature":      temperature,
        "use_chat_template": use_chat_template,
        "enable_thinking":  enable_thinking,
        "started_at":       datetime.now(timezone.utc).isoformat(),
    }

    stats     = LiveStats() if args.verbose else None
    open_mode = "a" if completed_ids else "w"
    n_done    = len(completed_ids)
    n_total   = len(completed_ids) + len(pending)

    with open(output_path, open_mode, encoding="utf-8") as out_f:
        pbar = tqdm(pending, desc=Path(args.model).name)
        for item in pbar:
            result = run_prompt(
                model, tokenizer,
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
                result_str = stats.update(item, result.get("response", ""))
                elapsed    = result.get("elapsed_s")
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
