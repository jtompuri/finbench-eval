#!/usr/bin/env python3
"""
Aggregate raw scored JSON files from results/raw/ into a tidy table.

Reads all score_*.json files, normalises them to a stable schema, and writes:
  - results/tidy/scores.csv
  - results/tidy/scores.parquet

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --raw-dir results/raw --out-dir results/tidy
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

MODEL_INFO = {
    # Local MLX models
    "poro8b": {
        "model_name": "Llama-Poro-2-8B IT",
        "model_family": "poro8b",
        "backend": "mlx_lm",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "gemma4_logprob": {
        "model_name": "Gemma 4 26B A4B (log-prob)",
        "model_family": "gemma4",
        "backend": "mlx_lm",
        "max_tokens": None,
        "temperature": None,
    },
    "gemma4e4b": {
        "model_name": "Gemma 4 E4B IT",
        "model_family": "gemma4e4b",
        "backend": "mlx_lm",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "gemma4e4b_think": {
        "model_name": "Gemma 4 E4B IT (thinking)",
        "model_family": "gemma4e4b",
        "backend": "mlx_lm",
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "gemma3": {
        "model_name": "Gemma 3 27B IT",
        "model_family": "gemma3",
        "backend": "mlx_lm",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "gemma4": {
        "model_name": "Gemma 4 26B A4B",
        "model_family": "gemma4",
        "backend": "mlx_lm",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    # Frontier API models — model_name is updated when the Phase 6 model ID is confirmed.
    # The key (e.g. "openai") must appear somewhere in the score filename stem
    # (e.g. score_openai_subset.json) for infer_model_key() to match it.
    "openai": {
        "model_name": "GPT-5.4",
        "model_family": "openai",
        "backend": "openai_api",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "openai_thinking": {
        "model_name": "GPT-5.4 (think)",
        "model_family": "openai",
        "backend": "openai_responses_api",
        "max_tokens": 2048,
        "temperature": 1.0,
    },
    "anthropic": {
        "model_name": "Claude Sonnet 4.6",
        "model_family": "anthropic",
        "backend": "anthropic_api",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "anthropic_thinking": {
        "model_name": "Claude Sonnet 4.6 (think)",
        "model_family": "anthropic",
        "backend": "anthropic_api",
        "max_tokens": 2048,
        "temperature": 1.0,
    },
    "google_flash": {
        "model_name": "Gemini 3 Flash Preview",
        "model_family": "google",
        "backend": "google_api",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "google_25pro": {
        "model_name": "Gemini 2.5 Pro",
        "model_family": "google",
        "backend": "google_api",
        "max_tokens": 8192,
        "temperature": 0.0,
    },
    "gemini31pro": {
        "model_name": "Gemini 3.1 Pro Preview",
        "model_family": "google",
        "backend": "openrouter",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "llama31": {
        "model_name": "Llama 3.1 8B IT",
        "model_family": "llama31",
        "backend": "mlx_lm",
        "max_tokens": 512,
        "temperature": 0.0,
    },
    "openai_temp1": {
        "model_name": "GPT-5.4 (t=1)",
        "model_family": "openai",
        "backend": "openai_api",
        "max_tokens": 512,
        "temperature": 1.0,
    },
    "google": {
        "model_name": "Gemini 3.1 Pro Preview",
        "model_family": "google",
        "backend": "google_api",
        "max_tokens": 512,
        "temperature": 0.0,
    },
}

TASK_GROUP = {
    # Comparable subset v1 (MCF)
    "arc_challenge_fi": "world_knowledge",
    "finbench_general_knowledge": "world_knowledge",
    "belebele_fin": "reading_comprehension",
    "squad_fi": "reading_comprehension",
    "goldenswag_fi": "commonsense",
    "scandisent_fi": "sentiment",
    "sib200_fi": "text_classification",
    # Extended subset — CF variants
    "arc_challenge_fi_cf": "world_knowledge",
    "finbench_gk_cf": "world_knowledge",
    "belebele_fin_cf": "reading_comprehension",
    "goldenswag_fi_cf": "commonsense",
    "scandisent_fi_cf": "sentiment",
    "sib200_fi_cf": "text_classification",
    # Extended subset — new tasks
    "truthfulqa_fi_mc1": "world_knowledge",
    "finbench_analogies": "relational_reasoning",
    "finbench_emotions": "sentiment",
    "finbench_hhh_alignment": "alignment",
    "finbench_similarities": "commonsense",
}

NORMALIZATION_VERSION = "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_model_key(stem: str) -> str:
    """Extract model key from filename stem, e.g. score_gemma3_subset -> gemma3.
    Longer/more specific keys are tried first to avoid false matches
    (e.g. 'google_flash' before 'google').
    """
    for key in sorted(MODEL_INFO, key=len, reverse=True):
        if key in stem:
            return key
    return "unknown"


def infer_subset(stem: str) -> str:
    """Infer subset label from filename stem."""
    if "extended" in stem:
        return "extended"
    if "subset" in stem:
        return "main_comparable"
    if "combined" in stem:
        return "main_comparable"
    if "smoke" in stem:
        return "smoke"
    return "unknown"


def rows_from_file(path: Path) -> list[dict]:
    """Read one score JSON file and return a list of tidy rows."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stem = path.stem  # e.g. score_gemma3_subset
    model_key = infer_model_key(stem)
    subset = infer_subset(stem)
    run_id = stem  # use filename stem as run_id

    info = MODEL_INFO.get(model_key, {
        "model_name": model_key,
        "model_family": model_key,
        "backend": "unknown",
        "max_tokens": None,
        "temperature": None,
    })

    per_task = data.get("per_task", {})
    if not per_task:
        warnings.warn(f"{path}: no per_task data found", stacklevel=2)
        return []

    rows = []
    for task, task_data in per_task.items():
        task_type = task_data.get("task_type", "gen")
        base = {
            "run_id": run_id,
            "model_family": info["model_family"],
            "model_name": info["model_name"],
            "task": task,
            "task_group": TASK_GROUP.get(task, "other"),
            "prompt_formulation": "p0",
            "prompt_variant": "p0",
            "subset": subset,
            "status": "ok",
            "notes": "",
            "backend": info["backend"],
            "temperature": info["temperature"],
            "max_tokens": info["max_tokens"],
            "normalization_version": NORMALIZATION_VERSION,
            "task_type": task_type,
            "n": task_data.get("n"),
        }

        if task_type in ("mcf_letter", "mcf_word"):
            ci = task_data.get("ci_95", [None, None])
            rows.append({
                **base,
                "metric_name": "accuracy",
                "metric_value": task_data.get("accuracy", task_data.get("score_avg")),
                "accuracy_all": task_data.get("accuracy_all"),
                "n_engaged": task_data.get("n_engaged"),
                "n_abstain": task_data.get("n_abstain"),
                "refusal_rate": task_data.get("refusal_rate"),
                "ci_lo": ci[0] if ci else None,
                "ci_hi": ci[1] if ci else None,
                "ci_method": "wilson",
                "ci_lo_alt": None,
                "ci_hi_alt": None,
            })
        else:  # gen
            ci_boot = task_data.get("ci_95", [None, None])
            ci_norm = task_data.get("ci_95_normal", [None, None])
            rows.append({
                **base,
                "metric_name": "f1",
                "metric_value": task_data.get("f1_avg", task_data.get("score_avg")),
                "accuracy_all": None,
                "n_engaged": None,
                "n_abstain": None,
                "refusal_rate": None,
                "ci_lo": ci_boot[0] if ci_boot else None,
                "ci_hi": ci_boot[1] if ci_boot else None,
                "ci_method": "bootstrap_2000",
                "ci_lo_alt": ci_norm[0] if ci_norm else None,
                "ci_hi_alt": ci_norm[1] if ci_norm else None,
            })
            rows.append({
                **base,
                "metric_name": "exact_match",
                "metric_value": task_data.get("exact_match_rate", 0.0),
                "accuracy_all": None,
                "n_engaged": None,
                "n_abstain": None,
                "refusal_rate": None,
                "ci_lo": None,
                "ci_hi": None,
                "ci_method": None,
                "ci_lo_alt": None,
                "ci_hi_alt": None,
            })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate raw scored JSON into tidy table.")
    parser.add_argument("--raw-dir", default="results/raw", help="Directory with score JSON files")
    parser.add_argument("--out-dir", default="results/tidy", help="Output directory for tidy table")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"ERROR: raw directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(raw_dir.glob("score_*.json"))
    if not json_files:
        print(f"ERROR: no score_*.json files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for path in json_files:
        rows = rows_from_file(path)
        all_rows.extend(rows)
        print(f"  read {path.name} → {len(rows)} rows")

    if not all_rows:
        print("ERROR: no rows aggregated — check input files", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Stable column order
    col_order = [
        "run_id", "model_family", "model_name",
        "task", "task_group", "task_type",
        "prompt_formulation", "prompt_variant",
        "metric_name", "metric_value",
        "ci_lo", "ci_hi", "ci_method",
        "ci_lo_alt", "ci_hi_alt",
        "accuracy_all", "n_engaged", "n_abstain", "refusal_rate",
        "subset", "status", "notes",
        "backend", "temperature", "max_tokens",
        "normalization_version", "n",
    ]
    df = df[col_order]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scores.csv"
    parquet_path = out_dir / "scores.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"\nAggregated {len(all_rows)} rows from {len(json_files)} file(s).")
    print(f"  CSV:     {csv_path}")
    print(f"  Parquet: {parquet_path}")

    # Short summary
    summary = (
        df[df["metric_name"].isin(["accuracy", "f1"])]
        .groupby(["model_name", "subset"])["metric_value"]
        .mean()
        .round(4)
    )
    print("\nMean score by model/subset (accuracy + F1):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
