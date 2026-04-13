#!/usr/bin/env python3
"""
McNemar pairwise significance tests for FIN-bench model comparisons.

For MCF tasks (binary correct/wrong per item): uses McNemar's test.
For gen tasks (F1 scores): uses Wilcoxon signed-rank test.

Applies Benjamini-Hochberg FDR correction for multiple comparisons.

Usage:
    python scripts/mcnemar_test.py
    python scripts/mcnemar_test.py --models anthropic openai poro8b gemma4e4b
    python scripts/mcnemar_test.py --alpha 0.05 --output results/mcnemar_results.json
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path


def mcnemar_chi2(n_ab: int, n_ba: int) -> tuple[float, float]:
    """McNemar's test: chi² with continuity correction.
    n_ab = model A correct, model B wrong
    n_ba = model B correct, model A wrong
    Returns (chi2, p_approx)
    """
    if n_ab + n_ba == 0:
        return 0.0, 1.0
    # With continuity correction (Edwards):
    chi2 = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
    # Approximate p-value using chi-square distribution with df=1
    # Critical values: chi2=3.84 → p=0.05, chi2=6.63 → p=0.01, chi2=10.83 → p=0.001
    p = chi2_p_approx(chi2, df=1)
    return round(chi2, 4), round(p, 6)


def chi2_p_approx(chi2: float, df: int = 1) -> float:
    """Approximate p-value from chi-square statistic (df=1 only, tail area)."""
    # Use regularized incomplete gamma function approximation
    # For df=1: p = erfc(sqrt(chi2/2)) = 1 - erf(sqrt(chi2/2))
    if chi2 <= 0:
        return 1.0
    x = math.sqrt(chi2 / 2)
    # Complementary error function (erfc) approximation
    # Using Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))))
    return poly * math.exp(-x * x)


def wilcoxon_signed_rank(diffs: list) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired differences.
    Returns approximate (W+, p_approx) using normal approximation.
    """
    # Remove zeros
    nonzero = [d for d in diffs if abs(d) > 1e-10]
    n = len(nonzero)
    if n < 5:
        return 0.0, 1.0
    # Rank by absolute value
    ranked = sorted(enumerate(nonzero), key=lambda x: abs(x[1]))
    w_plus = 0.0
    for rank_idx, (_, diff) in enumerate(ranked):
        rank = rank_idx + 1
        if diff > 0:
            w_plus += rank
    # Expected value and variance under H0
    expected = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    if variance == 0:
        return w_plus, 1.0
    z = (w_plus - expected) / math.sqrt(variance)
    # Two-tailed p-value
    p = 2 * norm_sf(abs(z))
    return round(w_plus, 1), round(p, 6)


def norm_sf(z: float) -> float:
    """Approximate survival function (1 - CDF) of standard normal."""
    # erfc approximation
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))))
    return poly * math.exp(-z * z) / 2


def benjamini_hochberg(p_values: list, alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of booleans (reject H0)."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = (rank + 1) / n * alpha
        if p <= threshold:
            reject[orig_idx] = True
    # Make monotone: once we stop rejecting, all later ones also not rejected
    max_reject_rank = -1
    for rank, (orig_idx, p) in enumerate(indexed):
        if reject[orig_idx]:
            max_reject_rank = rank
    for rank, (orig_idx, p) in enumerate(indexed):
        if rank <= max_reject_rank:
            reject[orig_idx] = True
    return reject


def load_scored_items(model_key: str, raw_dir: Path) -> dict | None:
    """Load per-item scored data. Returns dict keyed by item id."""
    path = raw_dir / f"score_{model_key}_combined.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    return {item["id"]: item for item in items}


def main():
    parser = argparse.ArgumentParser(description="McNemar pairwise significance tests.")
    parser.add_argument("--raw-dir", default="results/raw",
                        help="Directory with score_*_combined.json files")
    parser.add_argument("--models", nargs="+",
                        default=["poro8b", "llama31", "gemma4e4b", "gemma4e4b_think",
                                 "gemma3", "gemma4", "openai", "anthropic",
                                 "gemini31pro", "google_flash",
                                 "openai_thinking", "anthropic_thinking"],
                        help="Model keys to include")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="FDR significance threshold (default: 0.05)")
    parser.add_argument("--output", default=None,
                        help="Optional output JSON file")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    MODEL_NAMES = {
        "poro8b": "Poro-8B",
        "gemma4e4b": "Gemma 4 E4B",
        "gemma4e4b_think": "Gemma 4 E4B (think)",
        "gemma3": "Gemma 3 27B",
        "gemma4": "Gemma 4 26B",
        "openai": "GPT-5.4",
        "anthropic": "Claude Sonnet 4.6",
        "google_flash": "Gemini 3 Flash",
    }

    # Load all available model data
    model_items = {}
    for key in args.models:
        items = load_scored_items(key, raw_dir)
        if items is None:
            print(f"  (skipping {key}: combined scores not found)")
            continue
        model_items[key] = items
        print(f"  Loaded {key}: {len(items)} items")

    if len(model_items) < 2:
        print("Need at least 2 models to compare.")
        return

    # Find common items across all loaded models
    all_ids = set.intersection(*[set(v.keys()) for v in model_items.values()])
    print(f"\n{len(all_ids)} items shared across all {len(model_items)} models")

    # Get all tasks
    first_model_items = list(model_items.values())[0]
    tasks = sorted(set(item["task"] for item in first_model_items.values()))
    mcf_tasks = [t for t in tasks
                 if first_model_items.get(list(all_ids)[0], {}).get("task_type") in (None,) or
                 any(first_model_items[i]["task"] == t and
                     first_model_items[i]["task_type"] in ("mcf_letter", "mcf_word")
                     for i in all_ids if first_model_items.get(i))]

    # Actually check task types properly
    task_types = {}
    for item_id in all_ids:
        item = first_model_items.get(item_id)
        if item:
            task_types[item["task"]] = item["task_type"]

    mcf_tasks = [t for t, tt in task_types.items() if tt in ("mcf_letter", "mcf_word")]
    gen_tasks = [t for t, tt in task_types.items() if tt == "gen"]

    # Collect all test results
    all_results = []

    print("\n" + "=" * 100)
    print("MCNEMAR TESTS (MCF tasks, binary correct/wrong)")
    print("=" * 100)

    for key1, key2 in combinations(list(model_items.keys()), 2):
        name1 = MODEL_NAMES.get(key1, key1)
        name2 = MODEL_NAMES.get(key2, key2)

        print(f"\n  {name1} vs {name2}:")
        pair_results = []

        for task in sorted(mcf_tasks):
            task_ids = [i for i in all_ids
                        if first_model_items.get(i, {}).get("task") == task]
            if not task_ids:
                continue

            correct1 = {i: model_items[key1].get(i, {}).get("correct", False) for i in task_ids}
            correct2 = {i: model_items[key2].get(i, {}).get("correct", False) for i in task_ids}

            # Count discordant pairs
            n_ab = sum(1 for i in task_ids if correct1[i] and not correct2[i])  # A right, B wrong
            n_ba = sum(1 for i in task_ids if not correct1[i] and correct2[i])  # B right, A wrong
            n_total = len(task_ids)

            chi2, p = mcnemar_chi2(n_ab, n_ba)
            direction = ">" if n_ab > n_ba else "<" if n_ba > n_ab else "="
            pair_results.append({
                "task": task,
                "n_ab": n_ab, "n_ba": n_ba, "n": n_total,
                "chi2": chi2, "p": p,
                "model1_better": n_ab > n_ba,
            })
            print(f"    {task:<35} n_ab={n_ab:3d}  n_ba={n_ba:3d}  chi2={chi2:.2f}  p={p:.4f}  ({name1} {direction} {name2})")

        # Apply BH correction
        p_values = [r["p"] for r in pair_results]
        reject = benjamini_hochberg(p_values, args.alpha)
        significant_count = sum(reject)
        print(f"  → {significant_count}/{len(pair_results)} tasks significant after BH(α={args.alpha})")
        for i, r in enumerate(pair_results):
            if reject[i]:
                winner = name1 if r["model1_better"] else name2
                loser = name2 if r["model1_better"] else name1
                print(f"    ✅ {r['task']}: {winner} > {loser} (p={r['p']:.4f})")

        all_results.append({
            "model1": key1, "model2": key2,
            "tests": pair_results,
            "significant_tasks": [pair_results[i]["task"] for i, reject_i in enumerate(reject) if reject_i],
        })

    print("\n" + "=" * 100)
    print("WILCOXON SIGNED-RANK TESTS (gen tasks, F1 scores)")
    print("=" * 100)

    for key1, key2 in combinations(list(model_items.keys()), 2):
        name1 = MODEL_NAMES.get(key1, key1)
        name2 = MODEL_NAMES.get(key2, key2)
        print(f"\n  {name1} vs {name2}:")

        for task in sorted(gen_tasks):
            task_ids = [i for i in all_ids
                        if first_model_items.get(i, {}).get("task") == task]
            if not task_ids:
                continue

            diffs = []
            for i in task_ids:
                f1_1 = model_items[key1].get(i, {}).get("f1", 0.0)
                f1_2 = model_items[key2].get(i, {}).get("f1", 0.0)
                diffs.append(f1_1 - f1_2)

            w_plus, p = wilcoxon_signed_rank(diffs)
            avg_diff = sum(diffs) / len(diffs)
            direction = ">" if avg_diff > 0 else "<"
            sig = " ✅ SIGNIFICANT" if p < args.alpha else ""
            print(f"    {task:<35} mean_diff={avg_diff:+.4f}  W+={w_plus}  p={p:.4f}{sig}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
