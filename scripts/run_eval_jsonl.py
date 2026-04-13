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

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_mlx_prompt import load_model, run_prompt

REPO_ROOT = Path(__file__).parent.parent
RUN_SETTINGS_PATH = REPO_ROOT / "configs" / "run_settings.yaml"


def load_run_settings() -> dict:
    with open(RUN_SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_max_tokens(settings: dict, model_key: str | None, override: int | None) -> int:
    if override is not None:
        return override
    max_tokens_cfg = settings.get("generation", {}).get("max_tokens", {})
    if isinstance(max_tokens_cfg, int):
        return max_tokens_cfg
    if model_key and model_key in max_tokens_cfg:
        return max_tokens_cfg[model_key]
    return max_tokens_cfg.get("default", 512)


def load_jsonl(path: str) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


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
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already present in the output file (resume interrupted run)")
    args = parser.parse_args()

    settings = load_run_settings()
    max_tokens = resolve_max_tokens(settings, args.model_key, args.max_tokens)
    temperature = args.temperature if args.temperature is not None \
        else settings.get("generation", {}).get("temperature", 0.0)

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

    # Resume: find already-completed item IDs and skip them
    completed_ids: set[str] = set()
    if args.resume and output_path.exists():
        for record in load_jsonl(str(output_path)):
            if "id" in record:
                completed_ids.add(record["id"])
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
    enable_thinking = args.enable_thinking
    run_meta = {
        "model": args.model,
        "model_key": args.model_key,
        "input": args.input,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_chat_template": use_chat_template,
        "enable_thinking": enable_thinking,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Append if resuming, otherwise overwrite
    open_mode = "a" if completed_ids else "w"
    with open(output_path, open_mode, encoding="utf-8") as out_f:
        for item in tqdm(pending, desc="Running prompts"):
            prompt = item.get("prompt", "")
            result = run_prompt(
                model, tokenizer,
                prompt=prompt,
                model_path=args.model,
                max_tokens=max_tokens,
                temperature=temperature,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            record = {**item, **result, "run_meta": run_meta}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    total = len(completed_ids) + len(pending)
    print(f"Saved {total} results to {output_path} ({len(pending)} new)")


if __name__ == "__main__":
    main()
