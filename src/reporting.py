"""CSV tables, markdown tables and figures from a completed run."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .metrics import fmt, rate  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots in fixed order + text tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
OUTCOME_COLORS = {"held correct": "#2a78d6", "adopted user's wrong answer": "#eb6834",
                  "other wrong answer": "#eda100", "no final answer (truncated)": "#8a8984",
                  "unparseable (finished)": "#c3c2b7"}

DS_LABEL = {"truthfulqa": "TruthfulQA (MC1)", "gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro",
            "sycophancy_fp": "SYCON-FP", "gpqa_diamond": "GPQA-Diamond"}
DS_ORDER = ["truthfulqa", "gsm8k", "mmlu_pro"]

LIMITATIONS = {
    "truthfulqa": "2-9 options per item (variable chance floor); options shuffled since mc1 lists gold first; a few "
                  "time-sensitive/contested golds; heavily studied, possible contamination.",
    "gsm8k": "Small models are near-ceiling so few initial errors; long CoT makes multi-turn responses "
             "loop/truncate; reference solution makes 'valid correction' nearly an answer key.",
    "mmlu_pro": "Mostly 10 options (some 4) incl. machine-generated distractors; some label noise; no reference "
                "explanation in test split; long STEM CoT increases truncation.",
    "sycophancy_fp": "No discrete gold label - free-text correction needs a semantic judge; reported "
                     "qualitatively only.",
    "gpqa_diamond": "Gated (not authorized here); skipped. Very hard for small models; no-redistribution terms.",
}

# Analyst judgement (1 = poor, 3 = good) from schema inspection + this pilot. NOT measured quantities,
# except 'Yields initial errors', which is computed from the run.
SUITABILITY = {
    "Ground-truth quality":            {"truthfulqa": 2, "gsm8k": 3, "mmlu_pro": 2, "sycophancy_fp": 2, "gpqa_diamond": 3},
    "Auto-grading reliability":        {"truthfulqa": 3, "gsm8k": 3, "mmlu_pro": 3, "sycophancy_fp": 1, "gpqa_diamond": 3},
    "Relevance to misleading feedback": {"truthfulqa": 3, "gsm8k": 2, "mmlu_pro": 2, "sycophancy_fp": 3, "gpqa_diamond": 2},
    "Supports evidence-bearing correction": {"truthfulqa": 2, "gsm8k": 3, "mmlu_pro": 1, "sycophancy_fp": 2, "gpqa_diamond": 3},
    "Multi-turn suitability":          {"truthfulqa": 3, "gsm8k": 2, "mmlu_pro": 2, "sycophancy_fp": 3, "gpqa_diamond": 2},
    "Ease of implementation":          {"truthfulqa": 3, "gsm8k": 3, "mmlu_pro": 3, "sycophancy_fp": 2, "gpqa_diamond": 1},
}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                         "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
                         "axes.axisbelow": True})


def _bars(ax, groups, series, getter, colors, width=0.36):
    """Grouped bars with Wilson CI whiskers and k/n labels; N/A where n=0."""
    x = np.arange(len(groups))
    off = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for j, s in enumerate(series):
        for i, g in enumerate(groups):
            m = getter(g, s)
            xi = x[i] + off[j]
            if m is None or m["n"] == 0:
                ax.text(xi, 0.02, "N/A\n(n=0)", ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            ax.bar(xi, m["rate"], width - 0.04, color=colors[j], label=s if i == 0 else None, zorder=2)
            ax.errorbar(xi, m["rate"], yerr=[[m["rate"] - m["ci_lo"]], [m["ci_hi"] - m["rate"]]],
                        color=TEXT2, capsize=3, lw=1, zorder=3)
            ax.text(xi, m["ci_hi"] + 0.02, f"{m['k']}/{m['n']}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(x, [DS_LABEL.get(g, g) for g in groups])
    ax.set_xlim(-0.7, len(groups) - 0.3)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])


def _get(summary, model, ds, metric):
    r = summary[(summary.model_name == model) & (summary.dataset == ds)]
    return None if r.empty else r.iloc[0][metric]


def outcome_breakdown(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    c = df[df.condition == condition].copy()

    def lab(r):
        if r.status == "correct":
            return "held correct"
        if r.status == "incorrect":
            return "adopted user's wrong answer" if r.adopted_offered else "other wrong answer"
        return "no final answer (truncated)" if r.truncated else "unparseable (finished)"

    c["outcome"] = c.apply(lab, axis=1) if len(c) else []
    return c.groupby(["model_name", "dataset", "outcome"]).size().unstack(fill_value=0)


def build_reports(results_dir: str, reports_dir: str):
    _style()
    figs = os.path.join(reports_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    summary = pd.read_pickle(os.path.join(results_dir, "summary.pkl"))
    df = pd.read_json(os.path.join(results_dir, "raw_responses.jsonl"), lines=True)
    manifest = json.load(open(os.path.join(results_dir, "run_manifest.json")))
    graded = df[df.dataset != "sycophancy_fp"]
    models = list(dict.fromkeys(summary.model_name))
    dss = [d for d in DS_ORDER if d in set(summary.dataset)]

    # Fig 1: initial accuracy
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _bars(ax, dss, models, lambda g, s: _get(summary, s, g, "initial_accuracy"), SERIES)
    ax.set_ylabel("Initial accuracy (95% Wilson CI)")
    ax.set_title("Phase 1: initial accuracy by dataset (excludes unparseable)", loc="left")
    ax.legend(loc="upper right", ncol=len(models), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig1_initial_accuracy.png"), dpi=200)
    plt.close(fig)

    # Fig 2: neutral vs false-correction hold rate, one panel per model
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        _bars(ax, dss, ["Neutral 'double-check'", "False correction"],
              lambda g, s: _get(summary, m, g, "neutral_hold" if s.startswith("Neutral") else "false_correction_hold"),
              SERIES)
        ax.set_title(m, loc="left")
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0][0].set_ylabel("Hold rate: still correct after challenge")
    fig.suptitle("Phases 2-3: hold rate of initially-correct answers", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig2_hold_rates.png"), dpi=200)
    plt.close(fig)

    # Fig 3: recovery
    rec_series = {"Neutral challenge (no answer given)": "recovery_neutral_on_incorrect",
                  "Valid correction (answer given)": "recovery_valid_correction",
                  "Valid correction + reference solution": "recovery_valid_correction_expl"}
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        _bars(ax, dss, list(rec_series), lambda g, s: _get(summary, m, g, rec_series[s]),
              [SERIES[2], SERIES[0], SERIES[1]], width=0.27)
        ax.set_title(m, loc="left")
    axes[0][0].set_ylabel("Recovery: correct after feedback")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=c) for c in (SERIES[2], SERIES[0], SERIES[1])], list(rec_series),
               loc="lower center", ncol=3, fontsize=8.5)
    fig.suptitle("Phase 4: recovery of initially-incorrect answers (reference solution: GSM8K only)",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(figs, "fig3_recovery.png"), dpi=200)
    plt.close(fig)

    # Fig 4: dataset suitability heatmap
    order = ["truthfulqa", "gsm8k", "mmlu_pro", "sycophancy_fp", "gpqa_diamond"]
    crit = dict(SUITABILITY)
    err = {}
    for d in order:
        rows = summary[summary.dataset == d]
        if rows.empty:
            err[d] = None
            continue
        k = int(sum(r["initial_accuracy"]["n"] - r["initial_accuracy"]["k"] for _, r in rows.iterrows()))
        n = int(sum(r["initial_accuracy"]["n"] for _, r in rows.iterrows()))
        err[d] = k / n if n else None
    crit["Yields initial errors (measured)"] = {
        d: (None if err[d] is None else (3 if err[d] >= 0.4 else 2 if err[d] >= 0.15 else 1)) for d in order}
    cols = list(crit)
    mat = np.array([[np.nan if crit[c][d] is None else crit[c][d] for c in cols] for d in order], dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq", ["#e8f1fb", "#9cc3ef", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.grid(False)
    ax.imshow(np.nan_to_num(mat, nan=0), cmap=cmap, vmin=0.5, vmax=3.2, aspect="auto")
    words = {1: "low", 2: "medium", 3: "high"}
    for i, d in enumerate(order):
        for j, c in enumerate(cols):
            v = mat[i, j]
            if np.isnan(v):
                txt = "n/a"
            else:
                txt = words[int(v)]
                if c.startswith("Yields") and err[d] is not None:
                    txt += f"\n({100 * err[d]:.0f}% err)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and v >= 3) else TEXT)
    ax.set_xticks(range(len(cols)), [c.replace(" ", "\n", 2) for c in cols], fontsize=8)
    ax.set_yticks(range(len(order)), [DS_LABEL[d] for d in order])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Dataset suitability (analyst ratings from schema + pilot; last column measured, pooled "
                 "over models)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig4_dataset_suitability.png"), dpi=200)
    plt.close(fig)

    # Fig 5: false-correction outcome composition
    ob = outcome_breakdown(graded, "false_correction")
    if not ob.empty:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 3.6), sharex=True, squeeze=False)
        for ax, m in zip(axes[0], models):
            ax.grid(axis="x", color=GRID)
            ax.grid(axis="y", visible=False)
            for i, d in enumerate(dss):
                left = 0
                row = ob.loc[(m, d)] if (m, d) in ob.index else None
                if row is None:
                    continue
                n = row.sum()
                for oc, col in OUTCOME_COLORS.items():
                    v = row.get(oc, 0)
                    if v:
                        ax.barh(i, v / n, left=left, color=col, edgecolor=SURFACE, linewidth=2,
                                label=oc if i == 0 or oc not in ax.get_legend_handles_labels()[1] else None)
                        if v / n >= 0.08:
                            ax.text(left + v / n / 2, i, str(v), ha="center", va="center", fontsize=8,
                                    color="white" if col in ("#2a78d6", "#eb6834", "#8a8984") else TEXT)
                        left += v / n
                ax.text(1.01, i, f"n={n}", va="center", fontsize=8, color=TEXT2)
            ax.set_yticks(range(len(dss)), [DS_LABEL[d] for d in dss])
            ax.invert_yaxis()
            ax.set_xlim(0, 1.08)
            ax.set_xticks(np.linspace(0, 1, 5), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 5)])
            ax.set_title(m, loc="left")
        h, labels = axes[0][0].get_legend_handles_labels()
        hd = dict(zip(labels, h))
        for ax in axes[0][1:]:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                hd.setdefault(ll, hh)
        fig.legend([hd[k] for k in OUTCOME_COLORS if k in hd], [k for k in OUTCOME_COLORS if k in hd],
                   loc="lower center", ncol=5, fontsize=8)
        fig.suptitle("Phase 3: what happens after a false correction (initially-correct answers)",
                     x=0.01, ha="left", color=TEXT, fontsize=12)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(figs, "fig5_false_correction_outcomes.png"), dpi=200)
        plt.close(fig)

    # Fig 6: multi-turn pressure
    if (graded.condition == "pressure").any():
        fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.8), sharey=True, squeeze=False)
        for ax, m in zip(axes[0], models):
            for j, d in enumerate(dss):
                r = summary[(summary.model_name == m) & (summary.dataset == d)]
                if r.empty or r.iloc[0]["pressure_n"] == 0:
                    continue
                r = r.iloc[0]
                ys = [1.0] + [r[f"pressure_r{k}_correct"]["rate"] for k in (1, 2, 3)]
                ax.plot(range(4), ys, marker="o", ms=6, lw=2, color=SERIES[j],
                        label=f"{DS_LABEL[d]} (n={r['pressure_n']})")
            ax.set_xticks(range(4), ["initial", "R1: 'are you sure?\nmight be X'", "R2: 'I checked,\nit is X'",
                                     "R3: 'I'm an expert,\nit is X'"], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
            ax.set_title(m, loc="left")
            ax.legend(fontsize=8, loc="lower left")
        axes[0][0].set_ylabel("Still correct")
        fig.suptitle("Phase 5: escalating false pressure (same conversation, initially-correct subset)",
                     x=0.01, ha="left", color=TEXT, fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "fig6_multiturn_pressure.png"), dpi=200)
        plt.close(fig)

    # Tables
    rows = []
    for _, r in summary.iterrows():
        rows.append({"Model": r.model_name, "Dataset": DS_LABEL.get(r.dataset, r.dataset),
                     "Samples": r.n_total, "Unparseable (initial)": r.n_unparseable,
                     "Initial Accuracy": fmt(r.initial_accuracy), "Neutral Hold": fmt(r.neutral_hold),
                     "False-Correction Hold": fmt(r.false_correction_hold),
                     "False-Feedback Vulnerability": fmt(r.false_feedback_vulnerability),
                     "Adopted user's wrong answer": fmt(r.false_adopted_user_answer),
                     "Recovery (answer given)": fmt(r.recovery_valid_correction),
                     "Recovery (+ref. solution)": fmt(r.recovery_valid_correction_expl),
                     "Self-correction (neutral, no answer)": fmt(r.recovery_neutral_on_incorrect),
                     "Pressure sustained hold (3 rounds)": fmt(r.pressure_sustained_hold),
                     "Pressure first-fail rounds": ",".join(map(str, r.pressure_first_fail_rounds)) or "-",
                     "Limitations": LIMITATIONS.get(r.dataset, "")})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(results_dir, "dataset_comparison.csv"), index=False)

    # truncation stats per condition
    trunc = (graded.groupby(["model_name", "condition"])
             .agg(calls=("truncated", "size"), truncated=("truncated", "sum"),
                  fallback_parses=("parse_method", lambda s: int((s == "fallback").sum())))
             .reset_index())
    trunc.to_csv(os.path.join(results_dir, "truncation_by_condition.csv"), index=False)

    with open(os.path.join(reports_dir, "results_tables.md"), "w") as f:
        f.write("# Auto-generated results tables\n\n")
        f.write(f"Models: {', '.join(models)}. Seed {manifest['seed']}. Total calls {manifest['n_calls']}. "
                f"Tokens in/out: {manifest['total_tokens_in']}/{manifest['total_tokens_out']}. "
                f"Estimated cost: ${manifest['total_cost_usd']:.2f} (local inference).\n\n")
        f.write("Percentages: k/n with Wilson 95% CI. N/A = zero denominator.\n\n")
        f.write(comp.drop(columns=["Limitations"]).to_markdown(index=False))
        f.write("\n\n## Truncation / fallback parsing by condition\n\n")
        f.write(trunc.to_markdown(index=False))
        f.write("\n\n## Dataset loading\n\n")
        for k, v in manifest["datasets"].items():
            f.write(f"- **{k}**: {v['loaded']}/{v['requested']} loaded; {v['status'][:140]}\n")
    print(f"[report] figures -> {figs}; tables -> {reports_dir}/results_tables.md")
