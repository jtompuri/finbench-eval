"""
Compute BERTScore for SQuAD FI responses across all 12 model configurations.

Uses bert-base-multilingual-cased, which handles Finnish morphology without
penalising grammatically correct inflection forms that differ from gold spans.

Output: results/bertscore_squad.json
"""

import json
import os
from pathlib import Path
from bert_score import score as bert_score

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "bertscore_squad.json"

# Map raw file → display name (must match Table 1 ordering)
MODEL_FILES = [
    ("score_google_flash_combined.json",       "Gemini 3 Flash"),
    ("score_gemini31pro_combined.json",         "Gemini 3.1 Pro"),
    ("score_openai_thinking_combined.json",     "GPT-5.4 (think)"),
    ("score_openai_combined.json",              "GPT-5.4"),
    ("score_anthropic_combined.json",           "Claude Sonnet 4.6"),
    ("score_anthropic_thinking_combined.json",  "Claude Sonnet 4.6 (think)"),
    ("score_gemma4_combined.json",              "Gemma 4 26B"),
    ("score_gemma3_combined.json",              "Gemma 3 27B"),
    ("score_gemma4e4b_think_combined.json",     "Gemma 4 E4B (think)"),
    ("score_gemma4e4b_combined.json",           "Gemma 4 E4B"),
    ("score_poro8b_combined.json",              "Poro-8B"),
    ("score_llama31_combined.json",             "Llama 3.1 8B"),
]

BERT_MODEL = "bert-base-multilingual-cased"


def load_squad_items(filepath):
    with open(filepath) as f:
        data = json.load(f)
    items = [i for i in data.get("items", []) if i.get("task") == "squad_fi"]
    return items


def main():
    results = {}

    for filename, display_name in MODEL_FILES:
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"  MISSING: {filename}")
            continue

        items = load_squad_items(filepath)
        if not items:
            print(f"  No SQuAD items in {filename}")
            continue

        refs = [item["expected"] for item in items]
        hyps = [item["response_final"] for item in items]
        token_f1s = [item["f1"] for item in items]

        print(f"Computing BERTScore for {display_name} ({len(items)} items)...")
        P, R, F1 = bert_score(
            hyps, refs,
            model_type=BERT_MODEL,
            lang="fi",
            verbose=False,
            device=None,  # auto
        )

        bert_f1_mean = float(F1.mean())
        bert_p_mean  = float(P.mean())
        bert_r_mean  = float(R.mean())
        token_f1_mean = sum(token_f1s) / len(token_f1s)

        results[display_name] = {
            "n": len(items),
            "token_f1_mean": round(token_f1_mean, 4),
            "bert_f1_mean":  round(bert_f1_mean, 4),
            "bert_p_mean":   round(bert_p_mean, 4),
            "bert_r_mean":   round(bert_r_mean, 4),
            "delta":         round(bert_f1_mean - token_f1_mean, 4),
        }
        print(f"  Token-F1: {token_f1_mean:.3f}  |  BERTScore F1: {bert_f1_mean:.3f}  |  Δ: {bert_f1_mean - token_f1_mean:+.3f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "bert_model": BERT_MODEL,
            "note": "BERTScore F1 (multilingual BERT) vs token-F1 for SQuAD FI",
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {OUTPUT_FILE}")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Model':<30} {'Token-F1':>9} {'BERT-F1':>9} {'Δ':>7}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<30} {r['token_f1_mean']:>9.3f} {r['bert_f1_mean']:>9.3f} {r['delta']:>+7.3f}")
    print("="*70)


if __name__ == "__main__":
    main()
