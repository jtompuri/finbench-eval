"""
Shared plotting style for FIN-bench-v2 evaluation figures.

Import this module before creating any figures to apply the shared style.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
}

plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Display name mapping — data name → short display name used in figures + text
# ---------------------------------------------------------------------------

DISPLAY_NAMES: dict[str, str] = {
    "Llama-Poro-2-8B IT":        "Poro-8B",
    "Llama 3.1 8B IT":           "Llama 3.1 8B",
    "Gemma 4 E4B IT":            "Gemma 4 E4B",
    "Gemma 4 E4B IT (thinking)": "Gemma 4 E4B (think)",
    "Gemma 3 27B IT":            "Gemma 3 27B",
    "Gemma 4 26B A4B":           "Gemma 4 26B",
    "Gemma 4 26B A4B (log-prob)":"Gemma 4 26B (log-prob)",
    "Gemini 3 Flash Preview":    "Gemini 3 Flash",
    "Gemini 3.1 Pro Preview":    "Gemini 3.1 Pro",
    "Claude Sonnet 4.6":         "Claude Sonnet 4.6",
    "Claude Sonnet 4.6 (think)": "Claude Sonnet 4.6 (think)",
    "GPT-5.4":                   "GPT-5.4",
    "GPT-5.4 (think)":           "GPT-5.4 (think)",
    "GPT-5.4 (t=1)":             "GPT-5.4 (t=1)",
}


# ---------------------------------------------------------------------------
# Model palette — stable across figures
# ---------------------------------------------------------------------------

# Ordered model list — local models first (small→large), then frontier
MODEL_ORDER = [
    # Local 8B
    "Poro-8B",
    "Llama 3.1 8B",
    "Gemma 4 E4B",
    "Gemma 4 E4B (think)",
    # Local 27B+
    "Gemma 3 27B",
    "Gemma 4 26B",
    # Frontier — no thinking
    "Claude Sonnet 4.6",
    "GPT-5.4",
    # Frontier — thinking
    "Claude Sonnet 4.6 (think)",
    "GPT-5.4 (think)",
    "Gemini 3 Flash",
    "Gemini 3.1 Pro",
]

# Colorblind-friendly palette (Okabe-Ito inspired)
_PALETTE = [
    "#D55E00",  # vermilion      — Poro-8B
    "#994F00",  # dark orange    — Llama 3.1 8B
    "#E69F00",  # amber          — Gemma 4 E4B
    "#F0C040",  # yellow         — Gemma 4 E4B (think)
    "#0072B2",  # blue           — Gemma 3 27B
    "#56B4E9",  # sky blue       — Gemma 4 26B
    "#CC79A7",  # mauve          — Claude
    "#009E73",  # teal           — GPT
    "#BB99CC",  # light mauve    — Claude (think)
    "#44BB99",  # light teal     — GPT (think)
    "#000000",  # black          — Gemini Flash
    "#444444",  # dark grey      — Gemini 3.1 Pro
]

MODEL_COLORS: dict[str, str] = {
    name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(MODEL_ORDER)
}

# Excluded control runs — neutral grey to avoid collision with main palette
MODEL_COLORS["Gemma 4 26B (log-prob)"] = "#AAAAAA"
MODEL_COLORS["GPT-5.4 (t=1)"]          = "#CCCCCC"


def model_color(model_name: str) -> str:
    """Return a stable color for a model name, assigning a new one if needed."""
    if model_name not in MODEL_COLORS:
        idx = len(MODEL_COLORS)
        MODEL_COLORS[model_name] = _PALETTE[idx % len(_PALETTE)]
    return MODEL_COLORS[model_name]


# ---------------------------------------------------------------------------
# Status colors for coverage plots
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    "ok": "#4CAF50",
    "excluded": "#FFA726",
    "failed": "#EF5350",
    "unsupported": "#BDBDBD",
    "missing": "#E0E0E0",
}


# ---------------------------------------------------------------------------
# Task display order
# ---------------------------------------------------------------------------

TASK_ORDER = [
    "arc_challenge_fi",
    "belebele_fin",
    "goldenswag_fi",
    "scandisent_fi",
    "sib200_fi",
    "finbench_general_knowledge",
    "squad_fi",
    "truthfulqa_fi_mc1",
    "finbench_analogies",
    "finbench_emotions",
    "finbench_hhh_alignment",
    "finbench_similarities",
]

TASK_LABELS = {
    "arc_challenge_fi": "ARC-Challenge (fi)",
    "belebele_fin": "Belebele (fi)",
    "goldenswag_fi": "GoldenSwag (fi)",
    "scandisent_fi": "ScandiSent (fi)",
    "sib200_fi": "SIB-200 (fi)",
    "finbench_general_knowledge": "FINBench GK",
    "squad_fi": "SQuAD (fi) F1",
    # Extended subset — MCF generative
    "truthfulqa_fi_mc1": "TruthfulQA mc1 (fi)",
    "finbench_analogies": "FINBench Analogies",
    "finbench_emotions": "FINBench Emotions",
    "finbench_hhh_alignment": "FINBench HHH",
    "finbench_similarities": "FINBench Similarities",
    # Extended subset — CF variants (F1)
    "arc_challenge_fi_cf": "ARC-Challenge CF (fi)",
    "belebele_fin_cf": "Belebele CF (fi)",
    "goldenswag_fi_cf": "GoldenSwag CF (fi)",
    "scandisent_fi_cf": "ScandiSent CF (fi)",
    "sib200_fi_cf": "SIB-200 CF (fi)",
    "finbench_gk_cf": "FINBench GK CF",
}

# Extended MCF tasks shown in Figure 6
EXTENDED_MCF_ORDER = [
    "truthfulqa_fi_mc1",
    "finbench_analogies",
    "finbench_emotions",
    "finbench_hhh_alignment",
    "finbench_similarities",
]

# Extended CF tasks (shown separately as a validation finding)
EXTENDED_CF_ORDER = [
    "arc_challenge_fi_cf",
    "belebele_fin_cf",
    "goldenswag_fi_cf",
    "scandisent_fi_cf",
    "sib200_fi_cf",
    "finbench_gk_cf",
]


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    """Save figure as both PNG and PDF. Returns list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    return paths
