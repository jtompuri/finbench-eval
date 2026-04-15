"""
Shared utilities for all JSONL runner scripts.

Imported by run_eval_jsonl.py, run_llama_jsonl.py, run_vllm_jsonl.py,
and run_frontier_jsonl.py to avoid code duplication.
"""
import json
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Thinking-mode constants (Gemma 4 / compatible models)
# ---------------------------------------------------------------------------

THINKING_DELIMITER = "<channel|>"
THINKING_PREFIX    = "<|channel>thought\n"

SUMMARY_EVERY = 50   # print running accuracy summary every N items in --verbose mode

REPO_ROOT         = Path(__file__).parent.parent
RUN_SETTINGS_PATH = REPO_ROOT / "configs" / "run_settings.yaml"


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def load_run_settings() -> dict:
    with open(RUN_SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_max_tokens(settings: dict, model_key: str | None, override: int | None) -> int:
    if override is not None:
        return override
    max_tokens_cfg = settings.get("generation", {}).get("max_tokens", {})
    if isinstance(max_tokens_cfg, int):
        return max_tokens_cfg
    if model_key and model_key in max_tokens_cfg:
        return max_tokens_cfg[model_key]
    return max_tokens_cfg.get("default", 512)


def resolve_temperature(settings: dict, override: float | None) -> float:
    if override is not None:
        return override
    return settings.get("generation", {}).get("temperature", 0.0)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Load a JSONL file, skipping blank lines and warning on malformed JSON."""
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"WARNING: Skipping malformed JSON on line {lineno} of {path}",
                    file=sys.stderr,
                )
    return items


def build_resume_set(output_path: Path) -> set[str]:
    """Return the set of item IDs already written to output_path, or empty set."""
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    for record in load_jsonl(str(output_path)):
        if "id" in record:
            completed.add(record["id"])
    return completed


# ---------------------------------------------------------------------------
# Thinking output parser
# ---------------------------------------------------------------------------

def parse_thinking(raw_response: str) -> tuple[str, str | None]:
    """
    Split a raw model output into (response, thinking).

    If the model produced a thinking block (THINKING_DELIMITER present),
    `thinking` is the text before the delimiter (with THINKING_PREFIX stripped).
    Otherwise `thinking` is None.

    The caller is responsible for running extract_final_answer() on the response
    if needed — this function only handles the delimiter-based split.
    """
    if THINKING_DELIMITER not in raw_response:
        return raw_response, None
    before, after = raw_response.split(THINKING_DELIMITER, 1)
    thinking = before.removeprefix(THINKING_PREFIX).strip()
    return after.strip(), thinking


# ---------------------------------------------------------------------------
# Live statistics
# ---------------------------------------------------------------------------

class LiveStats:
    """
    Tracks per-task running accuracy for MCF tasks during verbose runs.
    Printed at regular intervals when --verbose is active.
    """

    def __init__(self):
        self.by_task: dict[str, dict] = {}

    def update(self, item: dict, response: str) -> str:
        """
        Score one item inline and return a short result string for live display.
        Only MCF tasks are scored live; gen tasks show a response snippet only.
        """
        from normalize_answer import extract_mcf_letter, extract_mcf_word

        task      = item.get("task", "?")
        task_type = item.get("task_type", "gen")
        expected  = item.get("expected", "")

        if task not in self.by_task:
            self.by_task[task] = {"n": 0, "correct": 0, "task_type": task_type}
        entry = self.by_task[task]
        entry["n"] += 1

        if task_type == "mcf_letter":
            got     = extract_mcf_letter(response) or "?"
            correct = got == expected.upper()
            entry["correct"] += int(correct)
            return f"exp={expected}  got={got}  {'✓' if correct else '✗'}"

        if task_type == "mcf_word":
            choices = item.get("expected_choices", [])
            got     = extract_mcf_word(response, choices) or "?"
            correct = got == expected.lower()
            entry["correct"] += int(correct)
            return f"exp={expected}  got={got}  {'✓' if correct else '✗'}"

        # gen task — show snippet only
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
            lines.append(f"  MCF overall: {total_c}/{total_n}  ({100*total_c/total_n:.1f}%)")
        lines.append("─" * 60)
        return lines
