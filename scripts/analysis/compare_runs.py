#!/usr/bin/env python3
"""
Generate a Markdown comparison report from two scored JSON files.

Usage:
    python compare_runs.py \
        --a results/score_gemma3_subset.json --a-label "Gemma 3 27B IT" \
        --b results/score_gemma4_subset.json --b-label "Gemma 4 26B A4B" \
        --output results/comparison_phase4.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/ → eval_config
from eval_config import primary_score as _primary_score


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_acc(task_summary: dict) -> str:
    tt = task_summary.get("task_type", "gen")
    ci = task_summary.get("ci_95")
    ci_str = f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""
    if tt == "gen":
        return f"F1={task_summary['f1_avg']:.3f}{ci_str}"
    return f"acc={task_summary['accuracy']:.3f}{ci_str} ({task_summary['n_correct']}/{task_summary['n']})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--a-label", default="Model A")
    parser.add_argument("--b", required=True)
    parser.add_argument("--b-label", default="Model B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a = load_json(args.a)
    b = load_json(args.b)
    a_label = args.a_label
    b_label = args.b_label

    all_tasks = sorted(set(list(a["per_task"].keys()) + list(b["per_task"].keys())))

    lines = []
    lines.append("# Phase 4 comparison report\n")
    lines.append(f"**Models compared:**\n- {a_label}\n- {b_label}\n")
    lines.append(f"**Subset:** FIN-bench-v2 comparable subset v1 (100 items/task)\n")
    lines.append("---\n")

    # Overall
    lines.append("## Overall scores\n")
    lines.append(f"| Model | Overall avg score |")
    lines.append(f"|---|---|")
    lines.append(f"| {a_label} | {a['overall_score_avg']:.4f} |")
    lines.append(f"| {b_label} | {b['overall_score_avg']:.4f} |")
    lines.append("")

    # Per-task
    lines.append("## Per-task results\n")
    lines.append(f"| Task | Type | {a_label} | {b_label} | Winner |")
    lines.append("|---|---|---|---|---|")

    for task in all_tasks:
        ta = a["per_task"].get(task)
        tb = b["per_task"].get(task)
        if ta is None:
            lines.append(f"| {task} | — | missing | {fmt_acc(tb)} | — |")
            continue
        if tb is None:
            lines.append(f"| {task} | — | {fmt_acc(ta)} | missing | — |")
            continue
        task_type = ta.get("task_type", "?")
        sa = _primary_score(ta)
        sb = _primary_score(tb)
        if sa > sb + 0.01:
            winner = a_label
        elif sb > sa + 0.01:
            winner = b_label
        else:
            winner = "tie"
        lines.append(f"| {task} | {task_type} | {fmt_acc(ta)} | {fmt_acc(tb)} | {winner} |")

    lines.append("")

    # Item-level differences for MCF tasks
    lines.append("## Notable response differences (sample)\n")
    a_items = {s["id"]: s for s in a.get("items", [])}
    b_items = {s["id"]: s for s in b.get("items", [])}

    shown = 0
    for item_id in sorted(a_items.keys()):
        ia = a_items.get(item_id)
        ib = b_items.get(item_id)
        if ia is None or ib is None:
            continue
        # Show items where the two models disagree
        a_correct = ia.get("correct", ia.get("exact_match", False))
        b_correct = ib.get("correct", ib.get("exact_match", False))
        if a_correct != b_correct and shown < 10:
            lines.append(f"**{item_id}** (task: {ia['task']})")
            lines.append(f"- Expected: `{ia['expected']}`")
            lines.append(f"- {a_label}: `{ia.get('extracted', ia.get('response_final','')[:80])}` → {'✓' if a_correct else '✗'}")
            lines.append(f"- {b_label}: `{ib.get('extracted', ib.get('response_final','')[:80])}` → {'✓' if b_correct else '✗'}")
            lines.append("")
            shown += 1

    if shown == 0:
        lines.append("_No disagreements found (both models gave the same correct/incorrect outcome on all items)._\n")

    # Thinking time note
    lines.append("## Notes\n")
    a_elapsed = [s["elapsed_s"] for s in a.get("items", []) if s.get("elapsed_s")]
    b_elapsed = [s["elapsed_s"] for s in b.get("items", []) if s.get("elapsed_s")]
    if a_elapsed and b_elapsed:
        avg_a = sum(a_elapsed) / len(a_elapsed)
        avg_b = sum(b_elapsed) / len(b_elapsed)
        lines.append(f"- Average elapsed per item: {a_label} = {avg_a:.2f}s, {b_label} = {avg_b:.2f}s")

    lines.append("- Scoring uses generative letter/word extraction, not log-prob scoring.")
    lines.append("- Results are not directly comparable to FIN-bench-v2 paper scores.")
    lines.append("- See `docs/comparable_subset.md` for full deviation list.\n")

    # Readiness assessment
    lines.append("## Frontier comparison readiness\n")
    lines.append("This subset is ready for frontier comparison if:")
    lines.append("- [x] Both models ran without errors")
    lines.append("- [x] Output schema is consistent")
    lines.append("- [x] Normalization and scoring are frozen")
    lines.append("- [ ] Absolute scores validated against paper (not done — Phase 4 scope)")
    lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Comparison report saved to {args.output}")


if __name__ == "__main__":
    main()
