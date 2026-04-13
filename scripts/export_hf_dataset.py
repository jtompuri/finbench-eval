#!/usr/bin/env python3
"""
Export FIN-bench-eval scored outputs to Hugging Face Datasets format.

Merges model responses (outputs/*.jsonl) with scoring metadata
(results/raw/score_*.json) into a single JSONL per model, then
optionally uploads to HF Datasets.

Output schema per item:
{
    "id":               str,    e.g. "arc_challenge_fi_000"
    "task":             str,    e.g. "arc_challenge_fi"
    "task_type":        str,    "mcf_letter" | "mcf_word" | "gen"
    "model":            str,    display name, e.g. "Claude Sonnet 4.6"
    "prompt":           str,    exact prompt sent to the model
    "expected":         str,    gold answer
    "response":         str,    full raw model response
    "extracted":        str,    extracted answer (letter/word/span)
    "abstain":          bool,   True if model declined to answer
    "correct":          bool,   True if extracted == expected
    "score":            float,  1.0 / 0.0 (or F1 for SQuAD)
}

Usage:
    # Dry run — write JSONL files locally only
    python scripts/export_hf_dataset.py --out-dir hf_export/

    # Upload to HF (requires HF_TOKEN in .env)
    python scripts/export_hf_dataset.py --out-dir hf_export/ --push
    python scripts/export_hf_dataset.py --out-dir hf_export/ --push \
        --repo jtompuri/finbench-eval-outputs
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

MODEL_MAP = [
    # (output_jsonl,                        score_json,                              display_name,                   model_id)
    # output_jsonl may be None — prompts are then reconstructed from data/finbench_combined_v1.jsonl
    ("combined_google_flash.jsonl",         "score_google_flash_combined.json",      "Gemini 3 Flash",               "gemini-3-flash-preview"),
    (None,                                  "score_gemini31pro_combined.json",       "Gemini 3.1 Pro",               "gemini-3.1-pro-preview"),
    ("combined_openai_thinking.jsonl",      "score_openai_thinking_combined.json",   "GPT-5.4 (think)",              "gpt-5.4"),
    ("combined_openai.jsonl",               "score_openai_combined.json",            "GPT-5.4",                      "gpt-5.4-2026-03-05"),
    ("combined_anthropic.jsonl",            "score_anthropic_combined.json",         "Claude Sonnet 4.6",            "claude-sonnet-4-6"),
    ("combined_anthropic_thinking.jsonl",   "score_anthropic_thinking_combined.json","Claude Sonnet 4.6 (think)",    "claude-sonnet-4-6"),
    ("combined_gemma4.jsonl",               "score_gemma4_combined.json",            "Gemma 4 26B",                  "gemma-4-26b-it-4bit"),
    ("combined_gemma3.jsonl",               "score_gemma3_combined.json",            "Gemma 3 27B",                  "gemma-3-27b-it-qat-4bit"),
    ("combined_gemma4e4b_think.jsonl",      "score_gemma4e4b_think_combined.json",   "Gemma 4 E4B (think)",          "gemma-4-e4b-it-4bit"),
    ("combined_gemma4e4b.jsonl",            "score_gemma4e4b_combined.json",         "Gemma 4 E4B",                  "gemma-4-e4b-it-4bit"),
    ("combined_poro8b.jsonl",               "score_poro8b_combined.json",            "Poro-8B",                      "Llama-Poro-2-8B-Instruct-mlx-4Bit"),
    ("combined_llama31.jsonl",              "score_llama31_combined.json",           "Llama 3.1 8B",                 "Meta-Llama-3.1-8B-Instruct-4bit"),
]


def load_outputs(path: Path) -> dict:
    """Load output JSONL → dict keyed by item id."""
    items = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            items[item["id"]] = item
    return items


def load_scores(path: Path) -> dict:
    """Load scored JSON → dict of item-level scores keyed by item id."""
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item for item in data.get("items", [])}


def merge(output_item: dict, score_item: dict, display_name: str) -> dict:
    """Merge output and score dicts into the public export schema."""
    item_id = score_item.get("id", output_item.get("id", ""))
    return {
        "id":        item_id,
        "task":      score_item.get("task", output_item.get("task", "")),
        "task_type": score_item.get("task_type", output_item.get("task_type", "")),
        "model":     display_name,
        "prompt":    output_item.get("prompt", ""),
        "expected":  output_item.get("expected", score_item.get("expected", "")),
        "response":  output_item.get("response", score_item.get("response_raw", "")),
        "extracted": score_item.get("extracted", ""),
        "abstain":   score_item.get("abstain", False),
        "correct":   score_item.get("correct", False),
        "score":     score_item.get("score", 0.0),
    }


def load_data_prompts(root: Path, data_dir_override: Path = None) -> dict:
    """Load prompts from the canonical data JSONL, keyed by item id."""
    data_path = data_dir_override if data_dir_override else (root / "data" / "finbench_combined_v1.jsonl")
    if not data_path.exists():
        return {}
    items = {}
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            items[item["id"]] = item
    return items


def export_model(output_file, score_file: str,
                 display_name: str, out_dir: Path,
                 outputs_dir_override: Path = None,
                 scores_dir_override: Path = None,
                 data_dir_override: Path = None) -> Path:
    slug = display_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    out_path = out_dir / f"{slug}.jsonl"

    outputs_dir = outputs_dir_override or (ROOT / "outputs")
    scores_dir = scores_dir_override or (ROOT / "results" / "raw")
    # Data dir: use override if provided, else ROOT
    data_root = data_dir_override.parent if data_dir_override else ROOT

    sp = scores_dir / score_file
    if not sp.exists():
        print(f"  MISSING score: {sp}")
        return None

    scores = load_scores(sp)

    # Load outputs if available, else fall back to data prompts
    outputs = {}
    if output_file is not None:
        op = outputs_dir / output_file
        if op.exists():
            outputs = load_outputs(op)
        else:
            print(f"  NOTE: output file missing ({output_file}), reconstructing prompts from data")

    if not outputs:
        outputs = load_data_prompts(data_root, data_dir_override)
        if not outputs:
            print(f"  WARNING: no data prompts found for {display_name}")

    merged = []
    for item_id, score_item in scores.items():
        out_item = outputs.get(item_id, {})
        merged.append(merge(out_item, score_item, display_name))

    with open(out_path, "w") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  {display_name}: {len(merged)} items → {out_path.name}")
    return out_path


def push_to_hf(out_dir: Path, repo: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in environment")
        return

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    print(f"Uploading to: https://huggingface.co/datasets/{repo}")

    for jsonl_file in sorted(out_dir.glob("*.jsonl")):
        api.upload_file(
            path_or_fileobj=str(jsonl_file),
            path_in_repo=f"data/{jsonl_file.name}",
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"  Uploaded: {jsonl_file.name}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Export scored outputs to HF Datasets format"
    )
    parser.add_argument("--out-dir", default="hf_export",
                        help="Local output directory (default: hf_export/)")
    parser.add_argument("--outputs-dir", default=None,
                        help="Directory containing combined_*.jsonl files "
                             "(default: <script_root>/outputs/)")
    parser.add_argument("--scores-dir", default=None,
                        help="Directory containing score_*_combined.json files "
                             "(default: <script_root>/results/raw/)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing finbench_combined_v1.jsonl "
                             "(default: <script_root>/data/)")
    parser.add_argument("--push", action="store_true",
                        help="Upload to Hugging Face Datasets")
    parser.add_argument("--repo", default="jtompuri/finbench-eval-outputs",
                        help="HF dataset repo id (default: jtompuri/finbench-eval-outputs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir_override = Path(args.outputs_dir) if args.outputs_dir else None
    scores_dir_override = Path(args.scores_dir) if args.scores_dir else None
    data_dir_override = Path(args.data_dir) if args.data_dir else None

    print(f"Exporting {len(MODEL_MAP)} model configurations → {out_dir}/\n")
    for output_file, score_file, display_name, _ in MODEL_MAP:
        export_model(output_file, score_file, display_name, out_dir,
                     outputs_dir_override, scores_dir_override, data_dir_override)

    print(f"\nExport complete: {out_dir}/")

    if args.push:
        print()
        push_to_hf(out_dir, args.repo)


if __name__ == "__main__":
    main()
