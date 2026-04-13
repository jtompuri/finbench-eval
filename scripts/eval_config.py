#!/usr/bin/env python3
"""
Shared configuration and utility functions for FIN-bench evaluation scripts.

Provides BASELINES, TASK_DISPLAY, MODEL_DISPLAY, normalized(), primary_score(),
and primary_ci() so that these are defined in one place and imported elsewhere.

Do not import other project modules here (avoid circular dependencies).
"""

# Random-chance baseline per task
BASELINES = {
    "arc_challenge_fi": 0.25,
    "belebele_fin": 0.25,
    "goldenswag_fi": 0.25,
    "scandisent_fi": 0.500,  # 2 classes (positive/negative); 0.333 in kytoniemi2025 incorrect
    "sib200_fi": 0.143,      # 7 choices (1/7); 0.067 in kytoniemi2025 incorrect
    "finbench_general_knowledge": 0.139,  # weighted mean 1/n_choices across 70 items (4–13 choices)
    "squad_fi": 0.0,
    "truthfulqa_fi_mc1": 0.219,  # weighted mean 1/n_choices across 100 items (2–11 choices); 0.125 in kytoniemi2025 incorrect
    "finbench_analogies": 0.20,
    "finbench_emotions": 0.125,
    "finbench_hhh_alignment": 0.50,
    "finbench_similarities": 0.242,  # weighted mean 1/n_choices across 76 items (4–6 choices); 0.200 in kytoniemi2025 incorrect
}

# Display names for tasks in tables and reports
TASK_DISPLAY = {
    "arc_challenge_fi": "ARC Challenge (fi)",
    "belebele_fin": "Belebele (fin)",
    "goldenswag_fi": "GoldenSwag (fi)",
    "scandisent_fi": "ScandiSent (fi)",
    "sib200_fi": "SIB-200 (fi)",
    "finbench_general_knowledge": "General Knowledge",
    "squad_fi": "SQuAD (fi) F1",
    "truthfulqa_fi_mc1": "TruthfulQA mc1",
    "finbench_analogies": "Analogies",
    "finbench_emotions": "Emotions",
    "finbench_hhh_alignment": "HHH Alignment",
    "finbench_similarities": "Similarities",
}

# Display names for models in tables and reports
MODEL_DISPLAY = {
    "poro8b": "Poro-8B",
    "llama31": "Llama 3.1 8B",
    "gemma4e4b": "Gemma 4 E4B",
    "gemma4e4b_think": "Gemma 4 E4B (think)",
    "gemma3": "Gemma 3 27B",
    "gemma4": "Gemma 4 26B",
    "openai": "GPT-5.4",
    "anthropic": "Claude Sonnet 4.6",
    "google_flash": "Gemini 3 Flash",
    "gemini31pro": "Gemini 3.1 Pro",
    "openai_thinking": "GPT-5.4 (think)",
    "anthropic_thinking": "Claude Sonnet 4.6 (think)",
    "openai_temp1": "GPT-5.4 (t=1)",
}


def normalized(score: float, task: str) -> float:
    """Return normalized score: (score - random_baseline) / (1 - random_baseline)."""
    b = BASELINES.get(task, 0.0)
    return (score - b) / (1.0 - b) if b < 1.0 else 0.0


def primary_score(task_data: dict) -> float:
    """Return the primary metric value for a task result dict.

    MCF tasks (mcf_letter, mcf_word): conditional accuracy (n_engaged denominator).
    Generative tasks: token F1 average.
    Falls back to score_avg if the preferred key is missing.
    """
    tt = task_data.get("task_type", "gen")
    if tt in ("mcf_letter", "mcf_word"):
        return task_data.get("accuracy", task_data.get("score_avg", 0.0))
    return task_data.get("f1_avg", task_data.get("score_avg", 0.0))


def primary_ci(task_data: dict) -> tuple:
    """Return (lo, hi) 95% confidence interval for the primary metric."""
    ci = task_data.get("ci_95", [0.0, 1.0])
    return (ci[0], ci[1])
