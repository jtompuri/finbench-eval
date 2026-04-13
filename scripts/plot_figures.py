#!/usr/bin/env python3
"""
Generate standard figures for the FIN-bench-v2 evaluation project.

Reads results/tidy/scores.csv and produces four figures in figures/:
  1. overall_model_comparison  — horizontal bar chart
  2. task_model_heatmap        — model × task grid
  3. prompt_variant_sensitivity — per-task scores by prompt variant
  4. coverage_summary          — stacked bar of ok/excluded/failed/missing

All figures saved as PNG and PDF.

Usage:
    python scripts/plot_figures.py
    python scripts/plot_figures.py --tidy results/tidy/scores.csv --out figures --subset main_comparable
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure scripts/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from eval_config import BASELINES, normalized as norm_score
from plot_style import (
    DISPLAY_NAMES,
    EXTENDED_CF_ORDER,
    EXTENDED_MCF_ORDER,
    MODEL_ORDER,
    STATUS_COLORS,
    TASK_LABELS,
    TASK_ORDER,
    model_color,
    save_figure,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "run_id", "model_name", "task", "metric_name", "metric_value", "subset", "status",
}


def load_tidy(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: tidy table not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERROR: tidy table missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    # Apply canonical display names (data names → short names used in figures and text)
    df["model_name"] = df["model_name"].map(lambda x: DISPLAY_NAMES.get(x, x))
    return df


def filter_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    filtered = df[df["subset"] == subset].copy()
    if filtered.empty:
        available = df["subset"].unique().tolist()
        print(f"WARNING: no rows for subset '{subset}'. Available: {available}", file=sys.stderr)
    return filtered


# ---------------------------------------------------------------------------
# Helper: ordered model list from data
# ---------------------------------------------------------------------------

def ordered_models(df: pd.DataFrame) -> list[str]:
    """Return models in MODEL_ORDER, appending any extra models found in data."""
    in_data = df["model_name"].unique().tolist()
    ordered = [m for m in MODEL_ORDER if m in in_data]
    ordered += [m for m in in_data if m not in ordered]
    return ordered


def ordered_tasks(df: pd.DataFrame) -> list[str]:
    """Return tasks in TASK_ORDER, appending any extra tasks found in data."""
    in_data = df["task"].unique().tolist()
    ordered = [t for t in TASK_ORDER if t in in_data]
    ordered += [t for t in in_data if t not in ordered]
    return ordered


# ---------------------------------------------------------------------------
# Figure 1 — overall_model_comparison
# ---------------------------------------------------------------------------

def plot_overall_model_comparison(df: pd.DataFrame, subset: str, out_dir: Path) -> list[Path]:
    """Horizontal bar chart of normalised mean score per model (SQuAD excluded)."""
    EXCLUDE_MODELS = {"Gemma 4 26B (log-prob)", "GPT-5.4 (t=1)"}
    EXCLUDE_TASKS  = {"squad_fi"}

    primary = df[
        (df["subset"] == subset) & (df["status"] == "ok") &
        (df["metric_name"].isin(["accuracy", "f1"])) &
        (~df["model_name"].isin(EXCLUDE_MODELS)) &
        (~df["task"].isin(EXCLUDE_TASKS)) &
        (df["task"].isin(BASELINES))          # only tasks with a defined baseline
    ].copy()

    if primary.empty:
        print("  WARNING: no data for overall_model_comparison — skipping")
        return []

    # Compute normalised score per row
    primary["norm"] = primary.apply(
        lambda r: norm_score(r["metric_value"], r["task"]), axis=1
    )

    grp = primary.groupby("model_name")["norm"]
    scores = grp.mean().rename("mean_score").reset_index()
    scores["se"] = grp.std().values / np.sqrt(grp.count().values)
    scores["ci"] = 1.96 * scores["se"]

    # Sort by score ascending so barh plots highest score at top
    scores = scores.sort_values("mean_score", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(scores))))

    bars = ax.barh(
        scores["model_name"],
        scores["mean_score"],
        xerr=scores["ci"],
        color=[model_color(m) for m in scores["model_name"]],
        error_kw={"ecolor": "#555555", "capsize": 4, "linewidth": 1.2},
        height=0.55,
        edgecolor="none",
    )

    # Value labels (offset past error bar)
    for bar, (_, row) in zip(bars.patches, scores.iterrows()):
        ax.text(
            row["mean_score"] + row["ci"] + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean_score']:.3f}", va="center", ha="left", fontsize=9,
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Normalised mean score (11 tasks, SQuAD excluded)")
    ax.set_title(f"Overall model comparison — {subset}")
    ax.set_ylabel("")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "overall_model_comparison")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 2 — task_model_heatmap
# ---------------------------------------------------------------------------

def plot_task_model_heatmap(df: pd.DataFrame, subset: str, out_dir: Path) -> list[Path]:
    """Heatmap of metric_value with rows=models, cols=tasks."""
    # Exclude validation/control runs and SQuAD from the main heatmap
    EXCLUDE_MODELS = {"Gemma 4 26B (log-prob)", "GPT-5.4 (t=1)"}
    EXCLUDE_TASKS = {"squad_fi"}

    primary = df[(df["subset"] == subset) & (df["metric_name"].isin(["accuracy", "f1"])) &
                 (~df["model_name"].isin(EXCLUDE_MODELS)) &
                 (~df["task"].isin(EXCLUDE_TASKS))]

    models = list(reversed(ordered_models(primary)))  # strongest at top
    tasks = [t for t in ordered_tasks(primary) if t not in EXCLUDE_TASKS]
    task_labels = [TASK_LABELS.get(t, t) for t in tasks]

    # Build matrix (NaN for missing)
    matrix = np.full((len(models), len(tasks)), np.nan)
    status_matrix = [["missing"] * len(tasks) for _ in range(len(models))]

    for ri, model in enumerate(models):
        for ci, task in enumerate(tasks):
            cell = primary[(primary["model_name"] == model) & (primary["task"] == task)]
            if not cell.empty:
                matrix[ri, ci] = cell["metric_value"].iloc[0]
                status_matrix[ri][ci] = cell["status"].iloc[0]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tasks)), max(3, 0.75 * len(models))))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#f0f0f0")

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Mark missing cells; choose text colour based on cell brightness
    for ri in range(len(models)):
        for ci in range(len(tasks)):
            s = status_matrix[ri][ci]
            if s == "missing":
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    fill=True, color="#cccccc", alpha=0.8, zorder=2,
                ))
                ax.text(ci, ri, "—", ha="center", va="center",
                        fontsize=8, color="#555555", zorder=3)
            elif not np.isnan(matrix[ri, ci]):
                val = matrix[ri, ci]
                # RdYlGn: dark ends, light middle — black text works throughout
                txt_color = "black"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color, zorder=3)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.7)
    ax.set_title(f"Per-task accuracy — {subset} (SQuAD F1 excluded)")
    ax.grid(False)

    # Only show legend if there are missing cells
    has_missing = any(
        status_matrix[ri][ci] == "missing"
        for ri in range(len(models))
        for ci in range(len(tasks))
    )
    if has_missing:
        patches = [mpatches.Patch(color="#cccccc", alpha=0.8, label="missing / not run")]
        ax.legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.9,
                  bbox_to_anchor=(0.0, -0.14))

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "task_model_heatmap")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 3 — prompt_variant_sensitivity
# ---------------------------------------------------------------------------

def plot_prompt_variant_sensitivity(out_dir: Path,
                                    data_path: Path = Path("results/raw/score_arc_variants_gemma4e4b.json"),
                                    ) -> list[Path]:
    """
    ARC Challenge FI engaged accuracy across five prompt variants (p0–p4) for Gemma 4 E4B.
    Data loaded from results/raw/score_arc_variants_gemma4e4b.json.
    Wilson 95% CIs computed from engaged items (abstentions excluded from denominator).

    This is the canonical Figure 3 in the paper (fig:sensitivity).
    """
    from collections import defaultdict

    if not data_path.exists():
        print(f"  WARNING: {data_path} not found — skipping prompt_variant_sensitivity")
        return []

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    if not items:
        print("  WARNING: no items in ARC variants file — skipping prompt_variant_sensitivity")
        return []

    # Group items by variant; ID format: arc_challenge_fi_NNN_pX
    variant_items: dict[str, list] = defaultdict(list)
    for item in items:
        parts = item["id"].rsplit("_", 1)
        if len(parts) == 2:
            variant_items[parts[1]].append(item)

    variants = sorted(variant_items.keys())

    # Compute engaged accuracy and Wilson CI per variant
    results = []
    for variant in variants:
        vitems = variant_items[variant]
        engaged = [it for it in vitems if not it.get("abstain", False)]
        n_eng = len(engaged)
        n_ok  = sum(1 for it in engaged if it.get("correct", False))
        acc = n_ok / n_eng if n_eng > 0 else 0.0
        lo, hi = wilson_ci(n_ok, n_eng)
        results.append({"variant": variant, "accuracy": acc, "ci_lo": lo, "ci_hi": hi, "n": n_eng})

    labels  = [r["variant"]  for r in results]
    scores  = np.array([r["accuracy"] for r in results])
    lo_errs = np.clip(scores - np.array([r["ci_lo"] for r in results]), 0, None)
    hi_errs = np.clip(np.array([r["ci_hi"] for r in results]) - scores, 0, None)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    color = model_color("Gemma 4 E4B")
    ax.bar(range(len(labels)), scores,
           color=color, alpha=0.85, width=0.55,
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(range(len(labels)), scores,
                yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=5, linewidth=1.2, zorder=4)

    # Value labels above bars
    for i, (score, hi) in enumerate(zip(scores, hi_errs)):
        ax.text(i, score + hi + 0.015, f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Engaged accuracy (Wilson 95% CI)")
    ax.set_xlabel("Prompt variant")
    ax.set_title("ARC Challenge FI — prompt variant sensitivity\n"
                 r"(Gemma 4 E4B, $n{\approx}100$ per variant)")
    ax.yaxis.grid(True, alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "arc_sensitivity")
    plt.close(fig)
    return paths


def plot_all_models_scatter(df: pd.DataFrame, subset: str, out_dir: Path,
                            fname: str = "all_models_scatter") -> list[Path]:
    """
    Diagnostic scatter: per-task scores for all models on a logit scale.
    Not used in the paper; kept as a diagnostic helper.
    """
    EXCLUDE_THINKING = {
        "Claude Sonnet 4.6 (think)",
        "GPT-5.4 (think)",
        "Gemma 4 E4B (think)",
        "Gemma 4 26B (log-prob)",
        "GPT-5.4 (t=1)",
    }

    primary = df[(df["subset"] == subset) & (df["status"] == "ok") &
                 (df["metric_name"].isin(["accuracy", "f1"])) &
                 (~df["model_name"].isin(EXCLUDE_THINKING))]

    if primary.empty:
        print(f"  WARNING: no data for {fname} — skipping")
        return []

    tasks = ordered_tasks(primary)
    models = ordered_models(primary)

    n_tasks = len(tasks)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * n_tasks), 9.0))

    x_base = np.arange(n_tasks)
    n_models = len(models)
    offsets = np.linspace(-0.25, 0.25, n_models) if n_models > 1 else [0.0]

    for mi, model in enumerate(models):
        mdata = primary[primary["model_name"] == model]
        y_vals = []
        for task in tasks:
            row = mdata[mdata["task"] == task]
            y_vals.append(row["metric_value"].iloc[0] if not row.empty else np.nan)

        y_plot = [np.clip(v, 0.01, 0.99) if not np.isnan(v) else np.nan for v in y_vals]

        ax.scatter(
            x_base + offsets[mi], y_plot,
            color=model_color(model), label=model,
            s=70, zorder=3, edgecolors="white", linewidths=0.5,
        )
        valid = [(x_base[i] + offsets[mi], v) for i, v in enumerate(y_plot) if not np.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=model_color(model), linewidth=1.0, alpha=0.5, zorder=2)

    import matplotlib.ticker as mticker
    ax.set_yscale("logit")
    ax.set_ylim(0.18, 0.995)
    logit_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ax.set_yticks(logit_ticks)
    ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2g}"))
    ax.yaxis.grid(True, alpha=0.3)

    ax.set_xticks(x_base)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (accuracy / F1) — logit scale")
    ax.set_title(f"Per-task scores — {subset} (logit scale)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, framealpha=0.9)
    ax.xaxis.grid(False)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.16)
    paths = save_figure(fig, out_dir, fname)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 4 — coverage_summary
# ---------------------------------------------------------------------------

def plot_coverage_summary(df: pd.DataFrame, subset: str, out_dir: Path) -> list[Path]:
    """
    Stacked bar showing ok / excluded / failed / unsupported / missing counts per model.
    Uses ALL tasks known from TASK_ORDER (plus any extras found in data) as the universe.
    """
    models = ordered_models(df[df["subset"] == subset]) if not df[df["subset"] == subset].empty \
        else ordered_models(df)

    # Universe of tasks visible in this subset
    tasks_in_subset = ordered_tasks(df[df["subset"] == subset]) if not df[df["subset"] == subset].empty \
        else TASK_ORDER
    n_tasks = len(tasks_in_subset)

    status_counts: dict[str, dict[str, int]] = {m: {s: 0 for s in STATUS_COLORS} for m in models}

    for model in models:
        mdf = df[(df["subset"] == subset) & (df["model_name"] == model) &
                 (df["metric_name"].isin(["accuracy", "f1"]))]
        seen_tasks = set()
        for _, row in mdf.iterrows():
            task = row["task"]
            seen_tasks.add(task)
            status_counts[model][row["status"]] += 1

        # Mark tasks not present at all as 'missing'
        for task in tasks_in_subset:
            if task not in seen_tasks:
                status_counts[model]["missing"] += 1

    fig, ax = plt.subplots(figsize=(max(5, 1.0 * len(models)), 4))

    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    status_order = ["ok", "excluded", "failed", "unsupported", "missing"]
    bars_legend = []

    for status in status_order:
        vals = np.array([status_counts[m].get(status, 0) for m in models], dtype=float)
        if vals.sum() == 0:
            continue
        b = ax.bar(
            x, vals, bottom=bottom,
            color=STATUS_COLORS[status], label=status,
            width=0.55, edgecolor="white", linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Number of tasks")
    ax.set_yticks(range(n_tasks + 1))
    ax.set_title(f"Task coverage per model — {subset}")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.grid(False)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "coverage_summary")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 5 — per_model_task_bars  (article-style, one panel per model)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def normal_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Normal-approx 95% CI for a list of continuous scores (e.g. F1).
    Falls back to Wilson CI when only one value is available (aggregate F1).
    """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (0.0, 0.0)
    mean = arr.mean()
    if len(arr) < 2:
        # Single aggregate value — use Wilson CI as approximation
        k = round(mean * 100)
        return wilson_ci(k, 100, z)
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return max(0.0, mean - z * se), min(1.0, mean + z * se)


def plot_per_model_task_bars(df: pd.DataFrame, subset: str, out_dir: Path) -> list[Path]:
    """
    One grouped bar figure per model — x=tasks, y=score, error bars = 95% CI.
    MCF tasks use Wilson CI; squad_fi uses normal-approx CI from per-item F1.
    All models combined into a single multi-panel figure (one row per model).
    Also saves individual per-model files for inclusion in papers.
    """
    primary = df[(df["subset"] == subset) & (df["metric_name"].isin(["accuracy", "f1"]))]
    if primary.empty:
        print("  WARNING: no data for per_model_task_bars — skipping")
        return []

    models = ordered_models(primary)
    tasks = ordered_tasks(primary)
    task_labels = [TASK_LABELS.get(t, t) for t in tasks]
    n_tasks = len(tasks)
    written: list[Path] = []

    # ── combined multi-panel figure ──────────────────────────────────────────
    n_models = len(models)
    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(max(9, 1.3 * n_tasks), 2.8 * n_models),
        sharex=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdata = primary[primary["model_name"] == model]
        scores, lo_errs, hi_errs = [], [], []

        for task in tasks:
            row = mdata[mdata["task"] == task]
            if row.empty:
                scores.append(np.nan)
                lo_errs.append(0.0)
                hi_errs.append(0.0)
                continue

            val = row["metric_value"].iloc[0]
            scores.append(val)
            n = int(row["n"].iloc[0]) if "n" in row.columns and not pd.isna(row["n"].iloc[0]) else 100

            task_type = row["task_type"].iloc[0] if "task_type" in row.columns else "mcf_letter"
            if task_type == "gen":
                lo, hi = normal_ci([val])   # single-value fallback; best effort
            else:
                k = round(val * n)
                lo, hi = wilson_ci(k, n)

            lo_errs.append(val - lo)
            hi_errs.append(hi - val)

        x = np.arange(n_tasks)
        scores_arr = np.array(scores, dtype=float)
        lo_arr = np.clip(lo_errs, 0, None)
        hi_arr = np.clip(hi_errs, 0, None)
        valid = ~np.isnan(scores_arr)

        color = model_color(model)
        ax.bar(x[valid], scores_arr[valid], color=color, alpha=0.85, width=0.6,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(x[valid], scores_arr[valid],
                    yerr=[lo_arr[valid], hi_arr[valid]],
                    fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)

        # Value labels on bars
        for xi in np.where(valid)[0]:
            val = scores_arr[xi]
            ax.text(xi, val + hi_arr[xi] + 0.025,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(model, fontsize=10, fontweight="bold", loc="left", pad=4)
        ax.yaxis.grid(True, alpha=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(np.arange(n_tasks))
    axes[-1].set_xticklabels(task_labels, rotation=30, ha="right", fontsize=9)

    fig.suptitle(
        f"Per-task scores with 95% CI — {subset}\n"
        "(MCF: Wilson CI · SQuAD: normal approx)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    written += save_figure(fig, out_dir, "per_model_task_bars")
    plt.close(fig)

    # ── individual per-model figures ─────────────────────────────────────────
    ind_dir = out_dir / "per_model"
    ind_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        mdata = primary[primary["model_name"] == model]
        scores, lo_errs, hi_errs = [], [], []

        for task in tasks:
            row = mdata[mdata["task"] == task]
            if row.empty:
                scores.append(np.nan); lo_errs.append(0.0); hi_errs.append(0.0)
                continue
            val = row["metric_value"].iloc[0]
            scores.append(val)
            n = int(row["n"].iloc[0]) if "n" in row.columns and not pd.isna(row["n"].iloc[0]) else 100
            task_type = row["task_type"].iloc[0] if "task_type" in row.columns else "mcf_letter"
            if task_type == "gen":
                lo, hi = normal_ci([val])
            else:
                k = round(val * n)
                lo, hi = wilson_ci(k, n)
            lo_errs.append(val - lo); hi_errs.append(hi - val)

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        x = np.arange(n_tasks)
        s_arr = np.array(scores, dtype=float)
        lo_a = np.clip(lo_errs, 0, None)
        hi_a = np.clip(hi_errs, 0, None)
        valid2 = ~np.isnan(s_arr)
        ax2.bar(x[valid2], s_arr[valid2], color=model_color(model), alpha=0.85, width=0.6,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax2.errorbar(x[valid2], s_arr[valid2], yerr=[lo_a[valid2], hi_a[valid2]],
                     fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)
        for xi in np.where(valid2)[0]:
            val = s_arr[xi]
            ax2.text(xi, val + hi_a[xi] + 0.025,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(task_labels, rotation=30, ha="right", fontsize=9)
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel("Score (accuracy / F1)")
        ax2.set_title(f"{model} — per-task scores with 95% CI", fontsize=10)
        ax2.yaxis.grid(True, alpha=0.5)
        ax2.xaxis.grid(False)
        ax2.set_axisbelow(True)
        fig2.tight_layout()

        safe_name = model.lower().replace(" ", "_").replace("(", "").replace(")", "")
        written += save_figure(fig2, ind_dir, f"tasks_{safe_name}")
        plt.close(fig2)

    return written


# ---------------------------------------------------------------------------
# Figure 6 — extended subset MCF results + CF validation panel
# ---------------------------------------------------------------------------

def _load_extended_ci(raw_dir: Path = Path("results/raw")) -> dict:
    """
    Load per-task CI bounds from all score_*_extended.json files.
    Returns {task: {"score": float, "ci_lo": float, "ci_hi": float, "model": str}}.
    """
    ci_data: dict = {}
    for path in sorted(raw_dir.glob("score_*extended*.json")):
        with open(path) as f:
            data = json.load(f)
        model_key = path.stem.replace("score_", "").replace("_extended", "")
        for task, td in data.get("per_task", {}).items():
            ci = td.get("ci_95", [td.get("score_avg", 0), td.get("score_avg", 0)])
            ci_data[task] = {
                "score": td.get("accuracy", td.get("f1_avg", td.get("score_avg", 0))),
                "ci_lo": ci[0],
                "ci_hi": ci[1],
                "model_key": model_key,
                "task_type": td.get("task_type", "gen"),
            }
    return ci_data


def plot_extended_results(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """
    Two-panel figure for the extended subset:
      Left:  MCF tasks — bar chart with Wilson 95% CI
      Right: CF tasks  — bar chart (low F1 validates MCF approach for instruction-tuned models)
    CI loaded from results/raw/score_*extended*.json.
    """
    ext_df = df[df["subset"] == "extended"].copy()
    if ext_df.empty:
        print("  No extended subset rows found — skipping Figure 6")
        return []

    ci_data = _load_extended_ci()
    models_present = ext_df["model_name"].unique().tolist()
    model_label = models_present[0] if models_present else "Gemma 4 E4B IT"

    def get_task_scores(task_list):
        rows = []
        for task in task_list:
            if task not in ci_data:
                continue
            td = ci_data[task]
            rows.append({
                "task": task,
                "label": TASK_LABELS.get(task, task),
                "model": model_label,
                "score": td["score"],
                "ci_lo": td["ci_lo"],
                "ci_hi": td["ci_hi"],
                "task_type": td["task_type"],
            })
        return rows

    mcf_rows = get_task_scores(EXTENDED_MCF_ORDER)
    cf_rows = get_task_scores(EXTENDED_CF_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Gemma 4 E4B — Extended Subset Results", fontsize=12, fontweight="bold")

    for ax, rows, title, ylabel in [
        (axes[0], mcf_rows, "MCF Generative Tasks (Accuracy)", "Accuracy"),
        (axes[1], cf_rows,  "CF Tasks (Token F1)\n— instruction-tuned models generate explanations", "Token F1"),
    ]:
        if not rows:
            ax.set_visible(False)
            continue

        labels = [r["label"] for r in rows]
        scores = np.array([r["score"] for r in rows])
        ci_lo  = np.array([r["ci_lo"] if not np.isnan(r["ci_lo"]) else r["score"] for r in rows])
        ci_hi  = np.array([r["ci_hi"] if not np.isnan(r["ci_hi"]) else r["score"] for r in rows])

        lo_errs = np.clip(scores - ci_lo, 0, None)
        hi_errs = np.clip(ci_hi - scores, 0, None)

        x = np.arange(len(labels))
        bars = ax.barh(x, scores, xerr=[lo_errs, hi_errs],
                       color=model_color(models_present[0]) if models_present else "#4C72B0",
                       alpha=0.82, capsize=4, error_kw={"elinewidth": 1.2})

        # Value labels
        for bar, score in zip(bars, scores):
            ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}", va="center", ha="left", fontsize=8)

        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axvline(x=0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", label="50% baseline")
        ax.invert_yaxis()

    plt.tight_layout()
    return save_figure(fig, out_dir, "extended_results")


# ---------------------------------------------------------------------------
# Figure 7 — Finnish specialisation: Poro-8B vs Llama 3.1 8B per-task delta
# ---------------------------------------------------------------------------

def plot_poro_vs_llama(df: pd.DataFrame, subset: str, out_dir: Path) -> list[Path]:
    """
    Horizontal bar chart of per-task score difference: Poro-8B minus Llama 3.1 8B IT.
    Positive bars = Finnish specialisation helps; negative = hurts.
    Error bars show 95% CI propagated as sqrt(se_poro^2 + se_llama^2).
    SQuAD excluded (F1 metric differs from accuracy tasks).
    """
    PORO  = "Poro-8B"
    LLAMA = "Llama 3.1 8B"
    EXCLUDE = {"squad_fi"}

    primary = df[
        (df["subset"] == subset) &
        (df["metric_name"].isin(["accuracy", "f1"])) &
        (~df["task"].isin(EXCLUDE))
    ]

    tasks = [t for t in ordered_tasks(primary) if t not in EXCLUDE]

    rows = []
    for task in tasks:
        p = primary[(primary["model_name"] == PORO)  & (primary["task"] == task)]
        l = primary[(primary["model_name"] == LLAMA) & (primary["task"] == task)]
        if p.empty or l.empty:
            continue
        pv = p["metric_value"].iloc[0]
        lv = l["metric_value"].iloc[0]
        # Normalised score difference: (poro_norm - llama_norm)
        # Because both use the same baseline b, the CI propagation is the same
        # as for raw differences scaled by 1/(1-b).
        b = BASELINES.get(task, 0.0)
        scale = 1.0 / (1.0 - b) if b < 1.0 else 1.0
        # half-widths of CIs (on raw scale, then scale to normalised)
        p_hi = p["ci_hi"].iloc[0] if "ci_hi" in p.columns and not pd.isna(p["ci_hi"].iloc[0]) else pv
        p_lo = p["ci_lo"].iloc[0] if "ci_lo" in p.columns and not pd.isna(p["ci_lo"].iloc[0]) else pv
        l_hi = l["ci_hi"].iloc[0] if "ci_hi" in l.columns and not pd.isna(l["ci_hi"].iloc[0]) else lv
        l_lo = l["ci_lo"].iloc[0] if "ci_lo" in l.columns and not pd.isna(l["ci_lo"].iloc[0]) else lv
        se_p = (p_hi - p_lo) / (2 * 1.96)
        se_l = (l_hi - l_lo) / (2 * 1.96)
        se_delta = ((se_p**2 + se_l**2) ** 0.5) * scale
        rows.append({
            "task": task,
            "label": TASK_LABELS.get(task, task),
            "delta": norm_score(pv, task) - norm_score(lv, task),
            "err": 1.96 * se_delta,
        })

    if not rows:
        print("  WARNING: no overlapping Poro/Llama data — skipping poro_vs_llama")
        return []

    rows = sorted(rows, key=lambda r: r["delta"])
    labels = [r["label"] for r in rows]
    deltas = np.array([r["delta"] for r in rows])
    errs   = np.array([r["err"]   for r in rows])

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.38 * len(rows))))

    colors = [model_color(PORO) if d >= 0 else "#aaaaaa" for d in deltas]
    ax.barh(labels, deltas, xerr=errs, color=colors, alpha=0.85,
            capsize=4, error_kw={"elinewidth": 1.2}, height=0.45,
            edgecolor="none")

    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Normalised score difference (Poro-8B − Llama 3.1 8B)")
    ax.set_title("Finnish specialisation effect per task\n"
                 "(positive = Poro-8B better; error bars = propagated 95% CI)")
    pad = 0.12
    ax.set_xlim(min(deltas - errs) - pad, max(deltas + errs) + pad)
    ax.xaxis.grid(True, alpha=0.5)
    ax.yaxis.grid(False)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, "poro_vs_llama")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate standard FIN-bench-v2 figures.")
    parser.add_argument("--tidy", default="results/tidy/scores.csv", help="Tidy CSV input")
    parser.add_argument("--out", default="figures", help="Output directory for figures")
    parser.add_argument("--subset", default="main_comparable", help="Subset to plot (default: main_comparable)")
    args = parser.parse_args()

    tidy_path = Path(args.tidy)
    out_dir = Path(args.out)
    subset = args.subset

    print(f"Reading tidy table: {tidy_path}")
    df = load_tidy(tidy_path)
    print(f"  {len(df)} rows, subsets: {df['subset'].unique().tolist()}")
    print(f"  models: {df['model_name'].unique().tolist()}")
    print(f"  tasks:  {df['task'].unique().tolist()}")
    print(f"Plotting subset: '{subset}'")
    print()

    written: list[Path] = []

    print("Figure 1: overall_model_comparison")
    written += plot_overall_model_comparison(df, subset, out_dir)

    print("Figure 2: task_model_heatmap")
    written += plot_task_model_heatmap(df, subset, out_dir)

    print("Figure 3: arc_sensitivity (ARC 5-variant bar chart for Gemma 4 E4B)")
    written += plot_prompt_variant_sensitivity(out_dir)

    print("Figure 4: coverage_summary")
    written += plot_coverage_summary(df, subset, out_dir)

    print("Figure 5: per_model_task_bars (with 95% CI)")
    written += plot_per_model_task_bars(df, subset, out_dir)

    print("Figure 6: extended_results (MCF + CF validation)")
    written += plot_extended_results(df, out_dir)

    print("Figure 7: poro_vs_llama (Finnish specialisation effect)")
    written += plot_poro_vs_llama(df, subset, out_dir)

    print(f"\nWrote {len(written)} file(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
