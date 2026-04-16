#!/usr/bin/env python3
"""
Score evaluation outputs from run_eval_jsonl.py.

Supports task-aware scoring for the FIN-bench-v2 comparable subset:
  - mcf_letter : extract first A/B/C/D letter and compare to expected
  - mcf_word   : check if expected word appears in response_final
  - gen        : exact_match (normalized) + token-level F1

Usage:
    python score_eval.py --input outputs/subset_gemma4.jsonl --output results/score_gemma4.json
"""
import argparse
import json
import math
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from normalize_answer import (
    best_f1_against_list,
    exact_match,
    extract_final_answer,
    extract_mcf_letter,
    extract_mcf_word,
    strip_markdown,
)


def wilson_ci(n_correct: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = n_correct / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def mean_se_ci(values: list, z: float = 1.96) -> tuple:
    """95% CI based on mean ± z * SE (normal approximation)."""
    n = len(values)
    if n < 2:
        return (0.0, 1.0)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(variance / n)
    return (round(max(0.0, mean - z * se), 4), round(min(1.0, mean + z * se), 4))


def bootstrap_ci(values: list, B: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple:
    """95% bootstrap percentile CI for the mean.

    Recommended for F1 scores which have a bimodal distribution
    (concentrated near 0 and 1) where normal approximation is inaccurate.
    """
    n = len(values)
    if n < 2:
        return (0.0, 1.0)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return (round(max(0.0, lo), 4), round(min(1.0, hi), 4))


def load_jsonl(path: str) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def score_item(item: dict) -> dict:
    task_type = item.get("task_type", "gen")
    expected = item.get("expected", "")
    expected_all = item.get("expected_all", [expected])
    expected_choices = item.get("expected_choices", [])
    response_raw = item.get("response", "")

    # Build cleaned response_final
    response_final = strip_markdown(extract_final_answer(response_raw)).strip()

    result = {
        "id": item.get("id"),
        "task": item.get("task"),
        "task_type": task_type,
        "expected": expected,
        "response_raw": response_raw,
        "response_final": response_final,
        "elapsed_s": item.get("elapsed_s"),
    }

    if task_type == "mcf_letter":
        extracted = extract_mcf_letter(response_raw)
        abstain  = (extracted == "")
        correct  = (not abstain) and (extracted == expected.upper())
        result.update({
            "extracted": extracted,
            "abstain": abstain,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
        })

    elif task_type == "mcf_word":
        extracted = extract_mcf_word(response_raw, expected_choices)
        abstain  = (extracted == "")
        correct  = (not abstain) and (extracted == expected.lower())
        result.update({
            "extracted": extracted,
            "abstain": abstain,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
        })

    else:  # gen
        em = any(exact_match(response_raw, ref, lowercase_flag=True, strip_punct=True)
                 for ref in expected_all)
        f1 = best_f1_against_list(response_raw, expected_all)
        result.update({
            "exact_match": em,
            "f1": round(f1, 4),
            "score": round(f1, 4),  # use F1 as primary gen score
        })

    return result


def summarise(scored: list) -> dict:
    """Build per-task and overall summaries."""
    by_task: dict = {}
    for s in scored:
        task = s.get("task", "unknown")
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(s)

    task_summaries = {}
    for task, items in by_task.items():
        n = len(items)
        task_type = items[0].get("task_type", "gen")
        scores = [s["score"] for s in items]
        avg = sum(scores) / n if n > 0 else 0.0

        task_summaries[task] = {
            "task_type": task_type,
            "n": n,
            "score_avg": round(avg, 4),
        }
        if task_type in ("mcf_letter", "mcf_word"):
            n_correct = sum(1 for s in items if s.get("correct", False))
            n_abstain = sum(1 for s in items if s.get("abstain", False))
            n_engaged = n - n_abstain
            # accuracy   = conditional on engagement (abstentions excluded from denominator)
            # accuracy_all = unconditional (abstentions counted as wrong) — kept for reference
            acc_engaged = round(n_correct / n_engaged, 4) if n_engaged > 0 else 0.0
            acc_all     = round(n_correct / n, 4) if n > 0 else 0.0
            ci_lo, ci_hi = wilson_ci(n_correct, n_engaged)
            task_summaries[task]["n_correct"]    = n_correct
            task_summaries[task]["n_abstain"]    = n_abstain
            task_summaries[task]["n_engaged"]    = n_engaged
            task_summaries[task]["refusal_rate"] = round(n_abstain / n, 4) if n > 0 else 0.0
            task_summaries[task]["accuracy"]     = acc_engaged   # primary metric
            task_summaries[task]["accuracy_all"] = acc_all       # reference (abstain = wrong)
            task_summaries[task]["ci_95"]        = [ci_lo, ci_hi]
        else:
            f1_values = [s["score"] for s in items]
            # Use bootstrap CI for F1: the distribution is bimodal (clustered near 0 and 1)
            # and normal approximation underestimates the true uncertainty.
            ci_lo, ci_hi = bootstrap_ci(f1_values)
            ci_lo_norm, ci_hi_norm = mean_se_ci(f1_values)  # kept as reference
            em_total = sum(1 for s in items if s.get("exact_match", False))
            task_summaries[task]["exact_match_count"] = em_total
            task_summaries[task]["exact_match_rate"] = round(em_total / n, 4) if n > 0 else 0.0
            task_summaries[task]["f1_avg"] = round(avg, 4)
            task_summaries[task]["ci_95"] = [ci_lo, ci_hi]           # bootstrap (primary)
            task_summaries[task]["ci_95_normal"] = [ci_lo_norm, ci_hi_norm]  # normal approx (reference)

    all_scores = [s["score"] for s in scored]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "total_items": len(scored),
        "overall_score_avg": round(overall_avg, 4),
        "note": "Smoke/comparable subset scoring only — not FIN-bench-v2 official scoring",
        "per_task": task_summaries,
        "items": scored,
    }


def main():
    parser = argparse.ArgumentParser(description="Score evaluation outputs.")
    parser.add_argument("--input", required=True, help="Input JSONL from run_eval_jsonl.py")
    parser.add_argument("--output", required=True, help="Output JSON file for scores")
    args = parser.parse_args()

    items = load_jsonl(args.input)
    scored = [score_item(item) for item in items]
    summary = summarise(scored)

    # Preserve run_meta from the source JSONL so that aggregate_results.py can
    # read the actual backend, max_tokens, and temperature instead of relying on
    # the hardcoded MODEL_INFO table (which only stores MLX defaults).
    run_meta = next(
        (item["run_meta"] for item in items if "run_meta" in item), None
    )
    if run_meta:
        summary["run_meta"] = run_meta

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Scored {len(scored)} items — overall avg score: {summary['overall_score_avg']:.4f}")
    for task, ts in summary["per_task"].items():
        if ts["task_type"] == "gen":
            print(f"  {task}: F1={ts['f1_avg']:.3f}  EM={ts['exact_match_rate']:.3f}  (n={ts['n']})")
        else:
            abstain_note = (f"  abstain={ts['n_abstain']}"
                            f"  refusal_rate={ts['refusal_rate']:.2f}"
                            if ts.get("n_abstain", 0) > 0 else "")
            print(f"  {task}: acc={ts['accuracy']:.3f}"
                  f"  ({ts['n_correct']}/{ts['n_engaged']}){abstain_note}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
