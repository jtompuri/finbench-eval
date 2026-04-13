#!/usr/bin/env python3
"""
Build the frozen FIN-bench-v2 comparable subset v1 JSONL.

Downloads a fixed first-N sample from each task's HuggingFace dataset,
formats prompts using the frozen p0 templates, and saves to data/finbench_subset_v1.jsonl.

See docs/comparable_subset.md for the full specification.

Usage:
    python build_subset_jsonl.py [--n-per-task 20] [--output data/finbench_subset_v1.jsonl]
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

LABEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def build_arc_challenge_fi(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-arc-c-fi-ht", split="test",
                      )
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        choices = doc["choices"]["text"]
        choices_str = "\n".join(f" {LABEL_LETTERS[j]} {t}" for j, t in enumerate(choices))
        prompt = (
            f"Mikä on paras vastaus kysymykseen {doc['question']}?\n"
            f"{choices_str}\n"
            f"Vastaus:"
        )
        items.append({
            "id": f"arc_challenge_fi_{i:03d}",
            "task": "arc_challenge_fi",
            "task_type": "mcf_letter",
            "prompt": prompt,
            "expected": doc["answerKey"],
            "expected_choices": list(LABEL_LETTERS[:len(choices)]),
        })
    return items


def build_belebele_fin(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-belebele-fi-og", split="test",
                      )
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        # Deviation from p0: use A/B/C/D labels instead of 1/2/3/4 for consistent extraction
        prompt = (
            f"Valitse tekstikatkelman perusteella oikea vastausvaihtoehto kysymykseen.\n\n"
            f"Teksti: {doc['flores_passage']}\n\n"
            f"Kysymys: {doc['question']}\n\n"
            f"Vastausvaihtoehdot:\n"
            f"A: {doc['mc_answer1']}\n"
            f"B: {doc['mc_answer2']}\n"
            f"C: {doc['mc_answer3']}\n"
            f"D: {doc['mc_answer4']}\n\n"
            f"Vastaus:"
        )
        expected = num_to_letter[str(doc["correct_answer_num"])]
        items.append({
            "id": f"belebele_fin_{i:03d}",
            "task": "belebele_fin",
            "task_type": "mcf_letter",
            "prompt": prompt,
            "expected": expected,
            "expected_choices": ["A", "B", "C", "D"],
        })
    return items


def _preprocess_goldenswag(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def build_goldenswag_fi(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-goldenswag-fi-ht", split="train",
                      )
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        ctx = doc["ctx_a"] + (" " + doc["ctx_b"].capitalize() if doc["ctx_b"] else "")
        query = _preprocess_goldenswag(doc["activity_label"] + ": " + ctx)
        endings = [_preprocess_goldenswag(e) for e in doc["endings"]]
        choices_str = "\n".join(f"{LABEL_LETTERS[j]}: {e}" for j, e in enumerate(endings))
        prompt = (
            f"{query}\n\n"
            f"Valitse seuraavista vaihtoehdoista loogisin jatko edelliselle tekstille.\n"
            f"{choices_str}\n\n"
            f"Vastaus:"
        )
        expected = LABEL_LETTERS[int(doc["label"])]
        items.append({
            "id": f"goldenswag_fi_{i:03d}",
            "task": "goldenswag_fi",
            "task_type": "mcf_letter",
            "prompt": prompt,
            "expected": expected,
            "expected_choices": list(LABEL_LETTERS[:len(endings)]),
        })
    return items


def build_scandisent_fi(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-scandisent-fi-mini", split="test",
                      )
    label_map = {"positive": "positiivinen", "negative": "negatiivinen"}
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        prompt = (
            f"Onko tekstissä esiintyvä tunne \"positiivinen\" vai \"negatiivinen\"?\n"
            f"Teksti: {doc['text']}\n"
            f"Tunne:"
        )
        expected = label_map[doc["label"]]
        items.append({
            "id": f"scandisent_fi_{i:03d}",
            "task": "scandisent_fi",
            "task_type": "mcf_word",
            "prompt": prompt,
            "expected": expected,
            "expected_choices": ["positiivinen", "negatiivinen"],
        })
    return items


def build_sib200_fi(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-sib-200-fi-og", split="test",
                      )
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        prompt = (
            f"Onko tekstin aihe \"politiikka\", \"viihde\", \"tiede/teknologia\", "
            f"\"urheilu\", \"matkailu\", \"terveys\" vai \"maantiede\"?\n"
            f"{doc['text']}\n"
        )
        expected = doc["choices"][doc["answer_idx"]]
        items.append({
            "id": f"sib200_fi_{i:03d}",
            "task": "sib200_fi",
            "task_type": "mcf_word",
            "prompt": prompt,
            "expected": expected,
            "expected_choices": doc["choices"],
        })
    return items


def build_finbench_general_knowledge(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-fbv1-stripped-fi-ht",
                      "general_knowledge_zero_shot", split="default",
                      )
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        choices_str = "\n".join(f" vaihtoehto: {s}" for s in doc["multiple_choice_targets"])
        prompt = f"{doc['query']}\n{choices_str}\nVastauksesi:"
        expected = doc["targets"][0]
        items.append({
            "id": f"finbench_gk_{i:03d}",
            "task": "finbench_general_knowledge",
            "task_type": "mcf_word",
            "prompt": prompt,
            "expected": expected,
            "expected_choices": doc["multiple_choice_targets"],
        })
    return items


def build_squad_fi(n: int) -> list:
    ds = load_dataset("TurkuNLP/finbenchv2-squad-strip-fi-mt", split="validation",
                      )
    items = []
    for i, doc in enumerate(ds.select(range(min(n, len(ds))))):
        prompt = (
            f"Otsikko: {doc['title']}\n\n"
            f"Teksti: {doc['context']}\n\n"
            f"Kysymys: {doc['question']}\n"
            f"Vastaus:"
        )
        expected_list = doc["answers"]["text"]
        items.append({
            "id": f"squad_fi_{i:03d}",
            "task": "squad_fi",
            "task_type": "gen",
            "prompt": prompt,
            "expected": expected_list[0] if expected_list else "",
            "expected_all": expected_list,
        })
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_BUILDERS = [
    build_arc_challenge_fi,
    build_belebele_fin,
    build_goldenswag_fi,
    build_scandisent_fi,
    build_sib200_fi,
    build_finbench_general_knowledge,
    build_squad_fi,
]


def main():
    parser = argparse.ArgumentParser(description="Build FIN-bench-v2 comparable subset JSONL.")
    parser.add_argument("--n-per-task", type=int, default=100)
    parser.add_argument("--output", default="data/finbench_subset_v1.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = []
    for builder in TASK_BUILDERS:
        task_name = builder.__name__.replace("build_", "")
        print(f"Building {task_name} ...")
        items = builder(args.n_per_task)
        all_items.extend(items)
        print(f"  -> {len(items)} items")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_items)} items saved to {output_path}")


if __name__ == "__main__":
    main()
