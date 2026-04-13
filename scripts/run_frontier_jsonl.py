#!/usr/bin/env python3
"""
Run a JSONL dataset through a frontier model API and save outputs as JSONL.

The output schema is identical to run_eval_jsonl.py so that score_eval.py,
aggregate_results.py, and compare_runs.py work without modification.

Usage — sequential (smoke test or small run):
    python run_frontier_jsonl.py \
        --provider anthropic --model-id claude-sonnet-4-6 \
        --input data/smoke_prompts.jsonl \
        --output outputs/smoke_anthropic.jsonl

Usage — sequential with live feedback:
    python run_frontier_jsonl.py \
        --provider anthropic --model-id claude-sonnet-4-6 \
        --input data/finbench_subset_v1.jsonl \
        --output outputs/subset_anthropic_seq.jsonl \
        --verbose

Usage — batch submit (OpenAI / Anthropic, ~50% discount, ~24 h turnaround):
    python run_frontier_jsonl.py \
        --provider openai --model-id gpt-5.4 \
        --input data/finbench_subset_v1.jsonl \
        --output outputs/subset_openai.jsonl \
        --batch

Usage — check batch status:
    python run_frontier_jsonl.py \
        --provider openai \
        --output outputs/subset_openai.jsonl \
        --batch-status

Usage — fetch completed batch:
    python run_frontier_jsonl.py \
        --provider openai --model-id gpt-5.4 \
        --input data/finbench_subset_v1.jsonl \
        --output outputs/subset_openai.jsonl \
        --batch-fetch

Usage — concurrent (Google, or any provider for faster sequential):
    python run_frontier_jsonl.py \
        --provider google --model-id gemini-3.1-pro-preview \
        --input data/finbench_subset_v1.jsonl \
        --output outputs/subset_google.jsonl \
        --concurrency 5

Usage — OpenRouter (Gemini 3.1 Pro Preview via OpenRouter, no daily quota):
    python run_frontier_jsonl.py \
        --provider openrouter \
        --model-id google/gemini-3.1-pro-preview-20260219 \
        --input data/finbench_subset_v1.jsonl \
        --output outputs/subset_openrouter_gemini31pro.jsonl \
        --concurrency 8

Notes:
- Batch metadata is stored alongside the output file as <stem>_batch_meta.json.
- --resume skips items whose id already appears in the output file.
- --batch is not supported for Google or OpenRouter; use --concurrency instead.
- Failed items (after retries) are written with response="" and an "error" field,
  so they can be identified and excluded from scoring if needed.
- Run from your own terminal with: source .env && caffeinate -i -s -w $$ &
  to prevent the Mac from sleeping during long runs.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import frontier_adapters as fa
from normalize_answer import extract_mcf_letter, extract_mcf_word


PROVIDERS = ("openai", "anthropic", "google", "openrouter",
             "openai-thinking", "anthropic-thinking")
SUMMARY_EVERY = 50   # print a running summary every N items in --verbose mode


# ---------------------------------------------------------------------------
# Live statistics tracker
# ---------------------------------------------------------------------------

class LiveStats:
    """
    Tracks per-task running accuracy for MCF tasks during sequential runs.
    Printed at regular intervals when --verbose is active.
    """
    def __init__(self):
        self.by_task: dict[str, dict] = {}

    def update(self, item: dict, response: str) -> str:
        """
        Score one item inline and return a short result string for live display.
        Only MCF tasks are scored live; gen tasks show response snippet only.
        """
        task = item.get("task", "?")
        task_type = item.get("task_type", "gen")
        expected = item.get("expected", "")

        if task not in self.by_task:
            self.by_task[task] = {"n": 0, "correct": 0, "task_type": task_type}
        entry = self.by_task[task]
        entry["n"] += 1

        if task_type == "mcf_letter":
            got = extract_mcf_letter(response) or "?"
            correct = got == expected.upper()
            entry["correct"] += int(correct)
            mark = "✓" if correct else "✗"
            return f"exp={expected}  got={got}  {mark}"

        elif task_type == "mcf_word":
            choices = item.get("expected_choices", [])
            got = extract_mcf_word(response, choices) or "?"
            correct = got == expected.lower()
            entry["correct"] += int(correct)
            mark = "✓" if correct else "✗"
            return f"exp={expected}  got={got}  {mark}"

        else:  # gen — just show snippet
            snippet = response[:50].replace("\n", " ")
            return f"→ {snippet!r}"

    def summary_lines(self) -> list[str]:
        lines = ["─" * 60, "  Running accuracy by task:"]
        total_n = total_c = 0
        for task, entry in sorted(self.by_task.items()):
            n, c = entry["n"], entry["correct"]
            if entry["task_type"] in ("mcf_letter", "mcf_word"):
                pct = 100 * c / n if n else 0
                lines.append(f"    {task:<35} {c}/{n}  ({pct:.1f}%)")
                total_n += n
                total_c += c
            else:
                lines.append(f"    {task:<35} {n} items  (gen — scored offline)")
        if total_n:
            overall = 100 * total_c / total_n
            lines.append(f"  MCF overall: {total_c}/{total_n}  ({overall:.1f}%)")
        lines.append("─" * 60)
        return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Load JSONL file, skipping any lines that cannot be parsed (e.g. truncated last line)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARNING: Skipping malformed JSON on line {lineno} of {path} "
                      f"(truncated write?): {line[:60]!r}", file=sys.stderr)
    return items


def batch_meta_path(output_path: Path) -> Path:
    """Derive the batch metadata file path from the output JSONL path."""
    return output_path.parent / f"{output_path.stem}_batch_meta.json"


def load_completed_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids = set()
    for rec in load_jsonl(str(output_path)):
        if "id" in rec:
            ids.add(rec["id"])
    return ids


def save_meta(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Batch metadata saved to {path}")


def load_meta(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_run_meta(provider: str, model_id: str, input_path: str,
                  max_tokens: int, temperature: float, mode: str) -> dict:
    return {
        "provider": provider,
        "model_id": model_id,
        "input": input_path,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def build_item_lookup(items: list) -> dict:
    """Build id→item dict for O(1) lookups when merging batch results."""
    return {it["id"]: it for it in items}


def _gen_kwargs_safe(max_tokens: int, temperature: float) -> dict:
    return {"max_tokens": max_tokens, "temperature": temperature}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a JSONL dataset through a frontier model API."
    )
    parser.add_argument("--provider", required=True, choices=PROVIDERS,
                        help="API provider: openai | anthropic | google")
    parser.add_argument("--model-id", default=None,
                        help="Model identifier as used by the provider's API "
                             "(e.g. gpt-5.4, claude-sonnet-4-6, gemini-3.1-pro-preview)")
    parser.add_argument("--input", default=None,
                        help="Input JSONL file (required except for --batch-status)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file (also determines batch-meta path)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens (default: 512 for OpenAI/Google, "
                             "1024 for Anthropic)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already present in the output file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-item response and running MCF accuracy. "
                             "Recommended for long terminal runs.")
    # Batch mode (OpenAI / Anthropic)
    parser.add_argument("--batch", action="store_true",
                        help="Submit batch job (OpenAI/Anthropic only)")
    parser.add_argument("--batch-status", action="store_true",
                        help="Print status of a previously submitted batch job")
    parser.add_argument("--batch-fetch", action="store_true",
                        help="Fetch and save results of a completed batch job")
    # Concurrent mode
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent API calls. Google requires this "
                             "instead of --batch. For OpenAI/Anthropic, prefer --batch "
                             "for cost savings; --concurrency is also accepted.")
    args = parser.parse_args()

    output_path = Path(args.output)
    meta_path = batch_meta_path(output_path)

    # Resolve max_tokens default per provider
    if args.provider in ("anthropic", "anthropic-thinking", "openai-thinking"):
        default_max_tokens = 2048
    else:
        default_max_tokens = 512

    max_tokens = args.max_tokens if args.max_tokens is not None else default_max_tokens
    temperature = args.temperature

    # -----------------------------------------------------------------------
    # Temperature safety check for thinking providers
    # The OpenAI Responses API and Anthropic extended thinking both require
    # temperature=1.0 and ignore any other value. Warn if the user passes
    # a different value so they are not surprised by the effective setting.
    # -----------------------------------------------------------------------
    THINKING_PROVIDERS = {"openai-thinking", "anthropic-thinking"}
    if args.provider in THINKING_PROVIDERS and temperature != 1.0:
        print(
            f"NOTE: {args.provider} requires temperature=1.0 "
            f"(you passed {temperature}). Overriding to 1.0.",
            file=sys.stderr,
        )
        temperature = 1.0

    # -----------------------------------------------------------------------
    # --batch-status: show status of existing batch
    # -----------------------------------------------------------------------
    if args.batch_status:
        if not meta_path.exists():
            print(f"ERROR: No batch metadata found at {meta_path}", file=sys.stderr)
            sys.exit(1)
        meta = load_meta(meta_path)
        provider = meta["provider"]   # use stored provider, not CLI arg
        print(f"Provider : {provider}")
        print(f"Batch ID : {meta['batch_id']}")
        print(f"Model    : {meta['model_id']}")
        print(f"Submitted: {meta['submitted_at']}")
        print(f"Items    : {meta['n_items']}")
        prov = fa.get_provider(provider)
        if not prov.supports_batch:
            print(f"ERROR: --batch-status not supported for provider '{provider}'",
                  file=sys.stderr)
            sys.exit(1)
        status = prov.poll_batch(meta["batch_id"])
        print(f"Status   : {status['status']}")
        print(f"Counts   : {status.get('request_counts', {})}")
        return

    # -----------------------------------------------------------------------
    # --batch-fetch: download completed batch results
    # -----------------------------------------------------------------------
    if args.batch_fetch:
        if not meta_path.exists():
            print(f"ERROR: No batch metadata found at {meta_path}", file=sys.stderr)
            sys.exit(1)
        if not args.input:
            print("ERROR: --input required for --batch-fetch", file=sys.stderr)
            sys.exit(1)
        if not args.model_id:
            print("ERROR: --model-id required for --batch-fetch", file=sys.stderr)
            sys.exit(1)

        meta = load_meta(meta_path)
        provider = meta["provider"]   # use stored provider
        items = load_jsonl(args.input)
        item_by_id = build_item_lookup(items)

        print(f"Fetching batch {meta['batch_id']} ...")
        prov = fa.get_provider(provider)
        if not prov.supports_batch:
            print(f"ERROR: --batch-fetch not supported for provider '{provider}'",
                  file=sys.stderr)
            sys.exit(1)
        status = prov.poll_batch(meta["batch_id"])
        if prov.batch_is_failed(status):
            print(f"ERROR: Batch ended with terminal status '{status['status']}' — "
                  f"it will never complete.", file=sys.stderr)
            print(f"Counts: {status.get('request_counts', {})}", file=sys.stderr)
            print("Delete the batch metadata file and re-submit to try again.",
                  file=sys.stderr)
            sys.exit(1)
        if not prov.batch_is_complete(status):
            print(f"Batch not yet complete. Status: {status['status']}")
            print(f"Counts: {status.get('request_counts', {})}")
            sys.exit(0)
        output_file_id = status.get("output_file_id")
        if provider == "openai" and not output_file_id:
            print("ERROR: Batch status is 'completed' but output_file_id is missing. "
                  "The batch may have failed entirely.", file=sys.stderr)
            sys.exit(1)
        rc = status.get("request_counts", {})
        if rc.get("succeeded", 1) == 0 and rc.get("errored", 0) > 0:
            print(f"WARNING: All {rc['errored']} batch items errored. "
                  f"Results will be empty.", file=sys.stderr)
        results = prov.fetch_batch(meta["batch_id"], output_file_id=output_file_id)

        run_meta = {
            "provider": provider,
            "model_id": args.model_id,
            "batch_id": meta["batch_id"],
            "mode": "batch_fetch",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            **meta.get("generation_kwargs", {}),
        }

        already_done = load_completed_ids(output_path) if args.resume else set()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        open_mode = "a" if already_done else "w"
        written = skipped_errors = 0
        with open(output_path, open_mode, encoding="utf-8") as f:
            for r in results:
                if r["id"] in already_done:
                    continue
                original = item_by_id.get(r["id"], {})
                record = {
                    **original,
                    "response": r["response"],
                    "elapsed_s": r["elapsed_s"],
                    "model": r.get("model") or args.model_id,
                    "generation_kwargs": meta.get("generation_kwargs", {}),
                    "run_meta": run_meta,
                }
                if r.get("error"):
                    record["error"] = r["error"]
                    skipped_errors += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        print(f"Written {written} records to {output_path}")
        if skipped_errors:
            print(f"  {skipped_errors} records have 'error' field — "
                  f"exclude from scoring with: "
                  f"grep -v '\"error\"' {output_path} > output_clean.jsonl")
        return

    # -----------------------------------------------------------------------
    # Require --input and --model-id for all remaining modes
    # -----------------------------------------------------------------------
    if not args.input:
        parser.error("--input is required")
    if not args.model_id:
        parser.error("--model-id is required")

    items = load_jsonl(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: filter already-completed items
    completed_ids = set()
    if args.resume and output_path.exists():
        completed_ids = load_completed_ids(output_path)
        print(f"Resuming: {len(completed_ids)} items already completed, skipping.")
    pending = [it for it in items if it.get("id") not in completed_ids]
    if len(pending) < len(items):
        print(f"Items to run: {len(pending)} / {len(items)}")

    if not pending:
        print("All items already completed — nothing to do.")
        return

    # -----------------------------------------------------------------------
    # --batch: submit batch job (OpenAI / Anthropic)
    # -----------------------------------------------------------------------
    if args.batch:
        prov = fa.get_provider(args.provider)
        if not prov.supports_batch:
            print(
                f"NOTE: {args.provider} has no batch endpoint. "
                "Use --concurrency 8 (or higher) for concurrent calls instead.",
                file=sys.stderr,
            )
            sys.exit(1)
        if meta_path.exists():
            print(f"ERROR: Batch metadata already exists at {meta_path}")
            print("  To fetch results: use --batch-fetch")
            print("  To start over:    delete the metadata file and re-run")
            sys.exit(1)

        meta = prov.call_batch_submit(pending, args.model_id, max_tokens, temperature)

        save_meta(meta, meta_path)
        print(f"\nNext steps:")
        print(f"  Check status : python scripts/run_frontier_jsonl.py "
              f"--provider {args.provider} --output {args.output} --batch-status")
        print(f"  Fetch results: python scripts/run_frontier_jsonl.py "
              f"--provider {args.provider} --model-id {args.model_id} "
              f"--input {args.input} --output {args.output} --batch-fetch")
        return

    # -----------------------------------------------------------------------
    # Concurrent mode (all providers, concurrency > 1)
    # Results are written to disk immediately as each future completes so that
    # a crash never loses more than one in-flight batch of requests.
    # -----------------------------------------------------------------------
    if args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        if args.provider not in ("google", "openrouter"):
            print(f"NOTE: --concurrency with {args.provider} runs concurrent "
                  f"API calls (no official batch endpoint used).")
        run_meta = make_run_meta(
            args.provider, args.model_id, args.input,
            max_tokens, temperature, f"concurrent/{args.concurrency}",
        )
        item_by_id = build_item_lookup(pending)

        # Build per-provider call function
        prov = fa.get_provider(args.provider)

        def _call_item(item):
            return prov.call_single(item["prompt"], args.model_id, max_tokens, temperature)

        gen_kw = _gen_kwargs_safe(max_tokens, temperature)

        def call_one(item):
            try:
                r = _call_item(item)
                return {**r, "id": item["id"], "error": None}
            except Exception as exc:
                return {
                    "id": item["id"], "response": "", "elapsed_s": None,
                    "model": args.model_id, "generation_kwargs": gen_kw,
                    "error": str(exc),
                }

        open_mode = "a" if completed_ids else "w"
        n_errors = 0
        n_written = 0

        with open(output_path, open_mode, encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_to_item = {
                    executor.submit(call_one, item): item for item in pending
                }
                for future in tqdm(
                    _as_completed(future_to_item),
                    total=len(pending),
                    desc=f"{args.provider}/{args.model_id} (concurrency={args.concurrency})",
                ):
                    r = future.result()
                    original = item_by_id.get(r["id"], {})
                    record = {**original, **r, "run_meta": run_meta}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    n_written += 1
                    if r.get("error"):
                        n_errors += 1

        total = len(completed_ids) + n_written
        print(f"Saved {total} results to {output_path} ({n_written} new)")
        if n_errors:
            print(f"  WARNING: {n_errors} items have errors — check 'error' field")
        return

    # -----------------------------------------------------------------------
    # Sequential single-call mode (default)
    # -----------------------------------------------------------------------
    run_meta = make_run_meta(
        args.provider, args.model_id, args.input,
        max_tokens, temperature, "sequential",
    )

    prov = fa.get_provider(args.provider)
    call_fn = lambda p: prov.call_single(p, args.model_id, max_tokens, temperature)

    stats = LiveStats() if args.verbose else None
    open_mode = "a" if completed_ids else "w"
    n_done = len(completed_ids)
    n_total = len(completed_ids) + len(pending)
    n_errors = 0

    with open(output_path, open_mode, encoding="utf-8") as out_f:
        pbar = tqdm(pending, desc=f"{args.provider} ({args.model_id})")
        for item in pbar:
            try:
                result = call_fn(item["prompt"])
                record = {**item, **result, "run_meta": run_meta}
                response = result.get("response", "")
                error = None
            except Exception as exc:
                print(f"\nFATAL error on {item.get('id', '?')}: {exc}", file=sys.stderr)
                response = ""
                error = str(exc)
                record = {
                    **item,
                    "response": response,
                    "elapsed_s": None,
                    "model": args.model_id,
                    "generation_kwargs": _gen_kwargs_safe(max_tokens, temperature),
                    "run_meta": run_meta,
                    "error": error,
                }
                n_errors += 1

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1

            if stats is not None:
                result_str = stats.update(item, response)
                elapsed = record.get("elapsed_s")
                elapsed_str = f"  {elapsed:.1f}s" if elapsed else ""
                tqdm.write(
                    f"[{n_done:>4}/{n_total}] {item.get('id', '?'):<40} "
                    f"{result_str}{elapsed_str}"
                    + (f"  ERROR: {error}" if error else "")
                )
                # Periodic summary every SUMMARY_EVERY items
                if n_done % SUMMARY_EVERY == 0:
                    for line in stats.summary_lines():
                        tqdm.write(line)

    print(f"Saved {n_done} results to {output_path} ({len(pending)} new)")
    if n_errors:
        print(f"  WARNING: {n_errors} items have errors — check 'error' field")

    # Final summary when verbose
    if stats is not None:
        print()
        for line in stats.summary_lines():
            print(line)


if __name__ == "__main__":
    main()
