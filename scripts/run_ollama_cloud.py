#!/usr/bin/env python3
"""
Run a JSONL dataset through Ollama Cloud and save outputs as JSONL.

Standalone, intentionally isolated from scripts/frontier_adapters.py and
scripts/run_frontier_jsonl.py. If we later decide to merge Ollama Cloud
support into the main pipeline, this script can be refactored into a
Provider entry in frontier_adapters.py. If not, removing this single
file is sufficient to drop support.

The output schema matches run_frontier_jsonl.py so that score_eval.py,
aggregate_results.py, and compare_runs.py work without modification.

Environment:
    OLLAMA_API_KEY   — Ollama Cloud API key (from https://ollama.com/settings/keys)

Usage:
    python scripts/run_ollama_cloud.py \\
        --model-id gemma3:27b \\
        --input data/finbench_combined_v1.jsonl \\
        --output outputs/combined_gemma4_31b_ollama.jsonl

    # Override base URL if Ollama changes its endpoint:
    python scripts/run_ollama_cloud.py \\
        --model-id gemma3:27b \\
        --base-url https://ollama.com/v1 \\
        --input ... --output ...
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Reuse existing helpers — no changes to runner_utils.py
sys.path.insert(0, str(Path(__file__).parent))
from runner_utils import (
    LiveStats,
    build_resume_set,
    load_jsonl,
)

DEFAULT_BASE_URL = "https://ollama.com/v1"


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable not set: {var}\n"
            f"Get a key at https://ollama.com/settings/keys and export it."
        )
    return val


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        code = int(status) if status is not None else 0
    except (TypeError, ValueError):
        code = 0
    if code in (401, 403, 404):
        return False
    if code in (429, 500, 502, 503, 529):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("timeout", "connection", "temporarily", "rate"))


def _retry(fn, retries: int = 3, backoff: float = 5.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"\nTransient error (attempt {attempt + 1}/{retries}): {exc}")
            print(f"Retrying in {wait:.0f}s ...")
            time.sleep(wait)


def run_ollama_prompt(
    client,
    prompt: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None = None,
) -> dict:
    def _call():
        t0 = time.time()
        kwargs = dict(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if reasoning_effort:
            # Ollama's OpenAI-compatible endpoint maps reasoning_effort
            # (high/medium/low/none) to its internal Think parameter.
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        completion = client.chat.completions.create(**kwargs)
        elapsed = round(time.time() - t0, 3)
        msg = completion.choices[0].message
        content = msg.content or ""
        # Ollama returns thinking in the `reasoning` attribute on the
        # message object (extra field, not in OpenAI spec). Merge into
        # response with <think>...</think> tags so downstream scoring
        # behaves consistently with other CoT runs in the pipeline.
        reasoning = getattr(msg, "reasoning", None) or ""
        if reasoning:
            response = f"<think>\n{reasoning}\n</think>\n\n{content}"
        else:
            response = content
        gen_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
        if reasoning_effort:
            gen_kwargs["reasoning_effort"] = reasoning_effort
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": gen_kwargs,
            "elapsed_s": elapsed,
        }

    return _retry(_call)


def main():
    parser = argparse.ArgumentParser(
        description="Run a JSONL dataset through Ollama Cloud."
    )
    parser.add_argument("--model-id", required=True,
                        help="Ollama Cloud model slug, e.g. 'gemma3:27b'")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file (supports resume)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Override OpenAI-compatible base URL "
                             f"(default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true",
                        help="Show running accuracy per task")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N pending items (smoke test)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests (Ollama Pro: 3, "
                             "Max: 10). Default: 1 (sequential).")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["high", "medium", "low", "none"],
                        help="Enable CoT/thinking on thinking-capable models. "
                             "Ollama maps this to its internal Think parameter. "
                             "Default: None (no thinking).")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(
        base_url=args.base_url,
        api_key=_require_env("OLLAMA_API_KEY"),
    )

    items = load_jsonl(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = build_resume_set(output_path)
    pending = [it for it in items if it.get("id") not in completed_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    run_meta = {
        "provider": "ollama-cloud",
        "model_id": args.model_id,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print(f"Ollama Cloud: {args.model_id}")
    print(f"  Input:  {args.input}  ({len(items)} items)")
    print(f"  Output: {args.output}  ({len(completed_ids)} already done, "
          f"{len(pending)} pending)")
    print(f"  Base URL: {args.base_url}")

    stats = LiveStats() if args.verbose else None
    open_mode = "a" if completed_ids else "w"
    n_errors = 0

    def _process(item: dict) -> tuple[dict, str, str | None]:
        try:
            result = run_ollama_prompt(
                client, item["prompt"], args.model_id,
                args.max_tokens, args.temperature,
                reasoning_effort=args.reasoning_effort,
            )
            return ({**item, **result, "run_meta": run_meta},
                    result.get("response", ""), None)
        except Exception as exc:
            record = {
                **item,
                "response": "",
                "elapsed_s": None,
                "model": args.model_id,
                "generation_kwargs": {
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                },
                "run_meta": run_meta,
                "error": str(exc),
            }
            return (record, "", str(exc))

    write_lock = threading.Lock()

    def _write(record: dict, response: str, err: str | None):
        nonlocal n_errors
        with write_lock:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            if err is not None:
                print(f"\nError on {record.get('id', '?')}: {err}",
                      file=sys.stderr)
                n_errors += 1
            if stats is not None:
                stats.update(record, response)

    with open(output_path, open_mode, encoding="utf-8") as out_f:
        if args.concurrency > 1:
            print(f"  Concurrency: {args.concurrency}")
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_process, it): it for it in pending}
                for fut in tqdm(as_completed(futures), total=len(pending),
                                desc=f"ollama-cloud ({args.model_id})"):
                    record, response, err = fut.result()
                    _write(record, response, err)
        else:
            for item in tqdm(pending,
                             desc=f"ollama-cloud ({args.model_id})"):
                record, response, err = _process(item)
                _write(record, response, err)

    print(f"\nDone. {len(pending) - n_errors}/{len(pending)} successful, "
          f"{n_errors} errors.")
    if stats is not None:
        for line in stats.summary_lines():
            print(line)


if __name__ == "__main__":
    main()
