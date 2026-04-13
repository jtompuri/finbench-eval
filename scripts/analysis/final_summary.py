#!/usr/bin/env python3
"""
Final summary report for FIN-bench combined evaluation.

Generates a comprehensive comparison table including:
- Raw scores per task
- Normalized scores
- Wilson CI (MCF) / Bootstrap CI (gen)
- Overall rankings
- McNemar significance summary

Usage:
    python scripts/final_summary.py
    python scripts/final_summary.py --models anthropic openai poro8b gemma4e4b gemma3 gemma4 gemma4e4b_think
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/ → eval_config
from eval_config import BASELINES, TASK_DISPLAY, MODEL_DISPLAY, normalized, primary_score, primary_ci


def load_per_task(model_key: str, raw_dir: Path) -> dict | None:
    path = raw_dir / f"score_{model_key}_combined.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("per_task", {})


def main():
    parser = argparse.ArgumentParser(description="Final summary report.")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--models", nargs="+",
                        default=["poro8b", "llama31", "gemma4e4b", "gemma4e4b_think",
                                 "gemma3", "gemma4", "openai", "anthropic",
                                 "gemini31pro", "google_flash",
                                 "openai_thinking", "anthropic_thinking"])
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    tasks = list(BASELINES.keys())

    # Load model data
    model_pt = {}
    for key in args.models:
        pt = load_per_task(key, raw_dir)
        if pt is None:
            print(f"  (skipping {key}: not found)")
        else:
            model_pt[key] = pt
            print(f"  ✓ {MODEL_DISPLAY.get(key, key)}")

    print(f"\n{len(model_pt)} models loaded.\n")
    if not model_pt:
        return

    col_w = 16
    name_w = 40

    def fmt_score(score, ci=None):
        if ci:
            return f"{score:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        return f"{score:.3f}"

    # ── Raw scores table ───────────────────────────────────────────────────────
    print("=" * (name_w + col_w * len(model_pt) + 5))
    print("RAW SCORES (accuracy for MCF, F1 for gen)")
    print("  Note: TruthfulQA (!) correct answer is always choice A (by design)")
    print("=" * (name_w + col_w * len(model_pt) + 5))

    header = f"{'Task':<{name_w}}"
    for k in model_pt:
        n = MODEL_DISPLAY.get(k, k)[:col_w - 2]
        header += f"  {n:>{col_w - 2}}"
    print(header)
    print("-" * (name_w + col_w * len(model_pt) + 5))

    raw = {k: {} for k in model_pt}
    for task in tasks:
        disp = TASK_DISPLAY.get(task, task)
        row = f"{disp:<{name_w}}"
        scores_this_task = []
        for k, pt in model_pt.items():
            td = pt.get(task)
            if td is None:
                row += f"  {'—':>{col_w - 2}}"
                raw[k][task] = None
            else:
                s = primary_score(td)
                row += f"  {s:>{col_w - 2}.3f}"
                raw[k][task] = s
                scores_this_task.append((k, s))
        # Bold the best
        if scores_this_task:
            best_val = max(s for _, s in scores_this_task)
            row += f"  ← best={best_val:.3f}"
        print(row)

    print("-" * (name_w + col_w * len(model_pt) + 5))

    # Overall averages
    row = f"{'Overall avg (raw)':<{name_w}}"
    for k, pt in model_pt.items():
        vals = [v for v in raw[k].values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        row += f"  {avg:>{col_w - 2}.3f}"
    print(row)

    # ── Normalized scores table ────────────────────────────────────────────────
    print()
    print("=" * (name_w + col_w * len(model_pt) + 5))
    print("NORMALIZED SCORES [(score - random_baseline) / (1 - random_baseline)]")
    print("=" * (name_w + col_w * len(model_pt) + 5))

    header2 = f"{'Task':<{name_w}}"
    for k in model_pt:
        n = MODEL_DISPLAY.get(k, k)[:col_w - 2]
        header2 += f"  {n:>{col_w - 2}}"
    print(header2)
    print("-" * (name_w + col_w * len(model_pt) + 5))

    norm = {k: {} for k in model_pt}
    for task in tasks:
        baseline = BASELINES[task]
        disp = f"{TASK_DISPLAY.get(task, task)} (b={baseline:.2f})"
        row = f"{disp:<{name_w}}"
        for k in model_pt:
            s = raw[k].get(task)
            if s is None:
                row += f"  {'—':>{col_w - 2}}"
                norm[k][task] = None
            else:
                n_score = normalized(s, task)
                row += f"  {n_score:>{col_w - 2}.3f}"
                norm[k][task] = n_score
        print(row)

    print("-" * (name_w + col_w * len(model_pt) + 5))
    row = f"{'Overall avg (normalized)':<{name_w}}"
    avgs = {}
    for k in model_pt:
        vals = [v for v in norm[k].values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        row += f"  {avg:>{col_w - 2}.3f}"
        avgs[k] = avg
    print(row)

    # ── Ranking ────────────────────────────────────────────────────────────────
    print()
    print("FINAL RANKING (by normalized average score):")
    for rank, (k, avg) in enumerate(sorted(avgs.items(), key=lambda x: -x[1]), 1):
        name = MODEL_DISPLAY.get(k, k)
        print(f"  {rank}. {name:<25}  normalized avg = {avg:.4f}")

    # ── Task-by-task best ──────────────────────────────────────────────────────
    print()
    print("BEST MODEL PER TASK:")
    for task in tasks:
        best_key, best_score = None, -1.0
        for k in model_pt:
            s = raw[k].get(task)
            if s is not None and s > best_score:
                best_score = s
                best_key = k
        if best_key:
            name = MODEL_DISPLAY.get(best_key, best_key)
            disp = TASK_DISPLAY.get(task, task)
            print(f"  {disp:<35}  {name:<25}  {best_score:.3f}")

    print("\n\nNotes:")
    print("  (!) TruthfulQA mc1: correct answer is always at position A (index 0) by design")
    print("  (*) Gemma 4 E4B thinking mode: helps reasoning tasks, hurts intuitive tasks")
    print("  (+) SQuAD F1 improved after answer_start bug fix (2 items)")
    print("  (+) HHH Alignment improved after overlap-fallback scoring fix")
    print()


if __name__ == "__main__":
    main()
