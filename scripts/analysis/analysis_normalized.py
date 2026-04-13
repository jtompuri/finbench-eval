#!/usr/bin/env python3
"""
Normalized score analysis for FIN-bench combined results.

Computes normalized scores: (score - random_baseline) / (1 - random_baseline)
for each task and each model, then aggregates to an overall normalized score.

Also computes Cohen's h effect sizes for pairwise model comparisons on MCF tasks.

Usage:
    python scripts/analysis_normalized.py
    python scripts/analysis_normalized.py --models anthropic openai poro8b gemma4e4b
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/ → eval_config
from eval_config import BASELINES, MODEL_DISPLAY, normalized, primary_score


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def load_scores(model_key: str, raw_dir: Path) -> dict | None:
    """Load the combined score file for a model key. Returns per_task dict."""
    path = raw_dir / f"score_{model_key}_combined.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_task", {})


def main():
    parser = argparse.ArgumentParser(description="Normalized score analysis.")
    parser.add_argument("--raw-dir", default="results/raw",
                        help="Directory with score_*_combined.json files")
    parser.add_argument("--models", nargs="+",
                        default=["poro8b", "llama31", "gemma4e4b", "gemma4e4b_think",
                                 "gemma3", "gemma4", "openai", "anthropic",
                                 "gemini31pro", "google_flash",
                                 "openai_thinking", "anthropic_thinking"],
                        help="Model keys to include")
    parser.add_argument("--output", default=None,
                        help="Optional output JSON file for results")
    parser.add_argument("--exclude-squad", action="store_true", default=True,
                        help="Exclude squad_fi from aggregate (default: True; use --no-exclude-squad to include)")
    parser.add_argument("--no-exclude-squad", dest="exclude_squad", action="store_false")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    all_tasks = list(BASELINES.keys())
    # squad_fi is excluded from aggregate by default (morphological artefact makes
    # it incomparable across model tiers; reported separately as a finding)
    tasks = [t for t in all_tasks if not (args.exclude_squad and t == "squad_fi")]
    squad_excluded = args.exclude_squad
    if squad_excluded:
        print("NOTE: squad_fi excluded from aggregate (morphological artefact). "
              "Use --no-exclude-squad to include.")

    # Load scores for all models
    model_scores = {}
    for key in args.models:
        pt = load_scores(key, raw_dir)
        if pt is None:
            print(f"  (skipping {key}: combined scores not found)")
            continue
        model_scores[key] = pt

    if not model_scores:
        print("No model scores found.")
        return

    print("\n" + "=" * 100)
    print("RAW SCORES (accuracy for MCF, F1 for gen)")
    print("=" * 100)

    # Print header
    header = f"{'Task':<35}"
    for key in model_scores:
        name = MODEL_DISPLAY.get(key, key)[:18]
        header += f"  {name:>18}"
    print(header)
    print("-" * (35 + 20 * len(model_scores)))

    raw_by_model_task = {}
    for task in tasks:
        row = f"{task:<35}"
        for key, pt in model_scores.items():
            task_data = pt.get(task)
            if task_data is None:
                score = None
                row += f"  {'N/A':>18}"
            else:
                score = primary_score(task_data)
                row += f"  {score:>18.3f}"
            if key not in raw_by_model_task:
                raw_by_model_task[key] = {}
            raw_by_model_task[key][task] = score
        print(row)

    # Print overall averages
    print("-" * (35 + 20 * len(model_scores)))
    row = f"{'Overall avg (raw)':<35}"
    for key in model_scores:
        scores = [v for v in raw_by_model_task[key].values() if v is not None]
        avg = sum(scores) / len(scores) if scores else 0.0
        row += f"  {avg:>18.3f}"
    print(row)

    # Normalized scores
    print("\n" + "=" * 100)
    print("NORMALIZED SCORES  [(score - random_baseline) / (1 - random_baseline)]")
    print("=" * 100)
    print(f"{'Task':<35}" + "".join(f"  {'Baseline':>10}" for _ in ["x"]) +
          "".join(f"  {MODEL_DISPLAY.get(k, k)[:18]:>18}" for k in model_scores))
    print("-" * (35 + 12 + 20 * len(model_scores)))

    norm_by_model_task = {}
    for task in tasks:
        baseline = BASELINES[task]
        row = f"{task:<35}  {baseline:>10.3f}"
        for key, pt in model_scores.items():
            raw = raw_by_model_task[key].get(task)
            if raw is None:
                norm = None
                row += f"  {'N/A':>18}"
            else:
                norm = normalized(raw, task)
                row += f"  {norm:>18.3f}"
            if key not in norm_by_model_task:
                norm_by_model_task[key] = {}
            norm_by_model_task[key][task] = norm
        print(row)

    print("-" * (35 + 12 + 20 * len(model_scores)))
    row = f"{'Overall avg (normalized)':<35}  {'':>10}"
    for key in model_scores:
        norms = [v for v in norm_by_model_task[key].values() if v is not None]
        avg = sum(norms) / len(norms) if norms else 0.0
        row += f"  {avg:>18.3f}"
    print(row)

    # Cohen's h pairwise comparison for MCF tasks
    print("\n" + "=" * 100)
    print("COHEN'S h EFFECT SIZE (MCF tasks, pairwise)")
    print("  |h| < 0.2 = small, 0.2–0.5 = medium, > 0.5 = large")
    print("=" * 100)

    mcf_tasks = [t for t in tasks if BASELINES.get(t, 0) > 0 and t != "squad_fi"]
    model_keys = list(model_scores.keys())

    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            key1, key2 = model_keys[i], model_keys[j]
            name1 = MODEL_DISPLAY.get(key1, key1)
            name2 = MODEL_DISPLAY.get(key2, key2)
            print(f"\n  {name1} vs {name2}:")
            for task in mcf_tasks:
                s1 = raw_by_model_task.get(key1, {}).get(task)
                s2 = raw_by_model_task.get(key2, {}).get(task)
                if s1 is None or s2 is None:
                    continue
                h = cohen_h(s1, s2)
                size = "large" if abs(h) > 0.5 else "medium" if abs(h) > 0.2 else "small"
                print(f"    {task:<35} h={h:+.3f} ({size})")

    # Save output if requested
    if args.output:
        output = {
            "squad_excluded_from_aggregate": squad_excluded,
            "n_tasks_in_aggregate": len(tasks),
            "raw_scores": raw_by_model_task,
            "normalized_scores": norm_by_model_task,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
