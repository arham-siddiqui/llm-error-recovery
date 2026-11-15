"""Paper figures for the v2 study (PDF + PNG). Fixed color semantics across every figure:
    neutral doubt  = gray (control, de-emphasized)     user asserts the CORRECT option = blue
    user asserts a WRONG option = orange                constant-intensity arm = dashed orange
Models are always separate panels/rows, never a color, so no figure has to be decoded with two color keys.
Palette = dataviz reference palette (validated: scripts/validate_palette.js, light mode).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis_v2 import DATASET_ORDER, MODEL_ORDER  # noqa: E402
from .metrics import wilson  # noqa: E402

BLUE, ORANGE, GRAY, VIOLET, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#6250d6", "#1baf7a"
TEXT, TEXT2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
COND_STYLE = {"neutral": ("Neutral doubt", GRAY), "assert_gold": ("User asserts correct option", BLUE),
              "assert_wrong": ("User asserts wrong option", ORANGE)}
DS_LABEL = {"truthfulqa_mc4": "TruthfulQA (4-option MC1)", "mmlu_pro_10": "MMLU-Pro (10-option)"}
MODEL_LABEL = {"llama3.2-3b-instruct": "Llama-3.2-3B", "qwen3-vl-4b-instruct": "Qwen3-VL-4B",
               "llama3.1-8b-instruct": "Llama-3.1-8B"}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.titlecolor": TEXT, "axes.titlesize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pct_axis(ax, top=1.0):
    ticks = np.linspace(0, 1, 6)
    ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
    ax.set_ylim(0, top)


def _models(df):
    return [m for m in MODEL_ORDER if m in set(df.model_name)]


def _datasets(df):
    return [d for d in DATASET_ORDER if d in set(df.dataset)]


def fig_initial_accuracy(cells: pd.DataFrame, out_dir: str):
    init = cells[cells.condition == "initial"]
    dss, models = _datasets(init), _models(init)
    fig, axes = plt.subplots(1, len(dss), figsize=(3.3 * len(dss), 2.8), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], dss):
        x = np.arange(len(models))
        for i, m in enumerate(models):
            r = init[(init.model_name == m) & (init.dataset == ds)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.bar(i, r.correct_rate, 0.55, color=VIOLET, zorder=2)
            ax.errorbar(i, r.correct_rate, yerr=[[r.correct_rate - r.correct_ci_lo], [r.correct_ci_hi - r.correct_rate]],
                        color=TEXT2, capsize=2.5, lw=0.9, zorder=3)
            ax.text(i, r.correct_ci_hi + 0.02, f"{r.correct_rate:.0%}\n{int(r.correct_k)}/{int(r.correct_n)}",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT)
            if r.n_initial_unparseable:
                ax.text(i, 0.02, f"{int(r.n_initial_unparseable)} no\nanswer", ha="center", va="bottom",
                        fontsize=6.5, color="white")
        ax.set_xticks(x, [MODEL_LABEL.get(m, m) for m in models], fontsize=8)
        ax.set_title(DS_LABEL.get(ds, ds))
        _pct_axis(ax, 1.12)
        chance = 0.25 if "truthfulqa" in ds else 0.10
        ax.axhline(chance, color=MUTED, lw=0.8, zorder=1)
        ax.text(len(models) - 0.5, chance + 0.01, "chance", fontsize=7, color=MUTED, ha="right", va="bottom")
    axes[0][0].set_ylabel("Initial accuracy (answered items)")
    _save(fig, out_dir, "fig1_initial_accuracy")


def fig_single_turn(cells: pd.DataFrame, out_dir: str):
    """P(final answer correct) after one follow-up, by condition, within initially-correct / -incorrect items."""
    dss, models = _datasets(cells), _models(cells)
    conds = ["neutral", "assert_gold", "assert_wrong"]
    fig, axes = plt.subplots(len(dss), 2, figsize=(7.2, 2.7 * len(dss)), sharey=True, squeeze=False)
    for i, ds in enumerate(dss):
        for j, grp in enumerate(["correct", "incorrect"]):
            ax = axes[i][j]
            x = np.arange(len(models))
            w = 0.26
            for k, c in enumerate(conds):
                for mi, m in enumerate(models):
                    r = cells[(cells.model_name == m) & (cells.dataset == ds) & (cells.initial_group == grp)
                              & (cells.condition == c)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    xi = mi + (k - 1) * w
                    ax.bar(xi, r.correct_rate, w - 0.03, color=COND_STYLE[c][1], zorder=2,
                           label=COND_STYLE[c][0] if mi == 0 else None)
                    ax.errorbar(xi, r.correct_rate, yerr=[[r.correct_rate - r.correct_ci_lo],
                                                          [r.correct_ci_hi - r.correct_rate]],
                                color=TEXT2, capsize=1.8, lw=0.8, zorder=3)
            for mi, m in enumerate(models):
                r = cells[(cells.model_name == m) & (cells.dataset == ds) & (cells.initial_group == grp)
                          & (cells.condition == "neutral")]
                if not r.empty:
                    ax.text(mi, -0.16, f"n={int(r.iloc[0].correct_n)}", ha="center", fontsize=7, color=MUTED,
                            transform=ax.get_xaxis_transform())
            ax.set_xticks(x, [MODEL_LABEL.get(m, m) for m in models], fontsize=8)
            _pct_axis(ax, 1.05)
            ax.set_title(f"{DS_LABEL.get(ds, ds)}\ninitially {grp} items")
            if j == 0:
                ax.set_ylabel("Final answer correct")
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.42), ncol=3, fontsize=8)
    fig.subplots_adjust(hspace=0.6)
    _save(fig, out_dir, "fig2_single_turn_conditions")


def fig_discrimination(eff: pd.DataFrame, out_dir: str):
    """Dumbbell: P(adopt asserted option | true) vs. P(adopt | false) on the same initially-incorrect items."""
    e = eff[eff.hypothesis == "H2"]
    dss, models = _datasets(e), _models(e)
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 0.55 * len(models) + 1.3), sharey=True,
                             squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for yi, m in enumerate(models):
            r = e[(e.model_name == m) & (e.dataset == ds)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = len(models) - 1 - yi
            for p, k, col in ((r.rate_a, r.k_a, ORANGE), (r.rate_b, r.k_b, BLUE)):
                lo, hi = wilson(int(k), int(r.n))[1:]
                ax.plot([lo, hi], [y, y], color=col, lw=1.0, alpha=0.6, zorder=2)
            ax.plot([r.rate_a, r.rate_b], [y, y], color=AXIS, lw=2, zorder=1)
            ax.scatter([r.rate_a], [y], s=36, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.2,
                       label="User asserts a wrong option" if yi == 0 else None)
            ax.scatter([r.rate_b], [y], s=36, color=BLUE, zorder=3, edgecolor="white", linewidth=1.2,
                       label="User asserts the correct option" if yi == 0 else None)
            ax.text(1.02, y, f"Δ {r['diff']:+.0%} [{r.ci_lo:+.0%}, {r.ci_hi:+.0%}]  n={int(r.n)}",
                    va="center", fontsize=7.5, color=TEXT, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(models)), [MODEL_LABEL.get(m, m) for m in reversed(models)])
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6), [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_title(DS_LABEL.get(ds, ds))
        ax.set_xlabel("Adopts the option the user asserted\n(initially incorrect items)")
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.42), ncol=2, fontsize=8)
    fig.subplots_adjust(wspace=0.75)
    _save(fig, out_dir, "fig3_discrimination")


def fig_outcomes(breakdown: pd.DataFrame, out_dir: str):
    """Stacked outcome shares for initially-correct items under the false assertion vs. neutral doubt."""
    b = breakdown
    dss, models = _datasets(b), _models(b)
    segs = [("held", "Kept correct answer", AQUA), ("adopted_user_option", "Switched to W (the wrong option the user asserts)", ORANGE),
            ("other_wrong", "Switched to another wrong option", VIOLET), ("no_answer", "No parseable answer", "#d9d8d2")]
    rows = [(ds, m, c) for ds in dss for m in models for c in ("neutral", "assert_wrong")]
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(rows) + 0.5 * len(dss) + 1.2))
    ylabels, ypos, y = [], [], 0.0
    for ds in dss:
        ax.text(-0.01, y, DS_LABEL.get(ds, ds), fontsize=8.5, color=TEXT, fontweight="bold",
                transform=ax.get_yaxis_transform(), ha="right", va="center")
        y += 1.0
        for m in models:
            for c in ("neutral", "assert_wrong"):
                r = b[(b.dataset == ds) & (b.model_name == m) & (b.condition == c)]
                ypos.append(y)
                ylabels.append(f"{MODEL_LABEL.get(m, m)} · {'neutral doubt' if c == 'neutral' else 'user asserts W'}")
                if not r.empty:
                    r = r.iloc[0]
                    left = 0.0
                    for key, _, col in segs:
                        v = r[key] / r.n
                        if v > 0:
                            ax.barh(y, v - 0.004, left=left, height=0.72, color=col, zorder=2,
                                    hatch="////" if key == "no_answer" else None, edgecolor="white", linewidth=0)
                            if v >= 0.06:
                                ax.text(left + v / 2, y, f"{r[key]}", ha="center", va="center", fontsize=6.5,
                                        color="white" if key != "no_answer" else TEXT)
                        left += v
                    ax.text(1.01, y, f"n={r.n}", va="center", fontsize=7, color=MUTED)
                y += 1
            y += 0.4
        y += 0.4
    ax.set_yticks(ypos, ylabels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of initially-correct items")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch="////" if k == "no_answer" else None) for k, _, c in segs]
    ax.legend(handles, [s[1] for s in segs], loc="upper center", bbox_to_anchor=(0.45, -0.1 - 1.5 / len(rows)),
              ncol=2, fontsize=7.5)
    _save(fig, out_dir, "fig4_false_assertion_outcomes")


def fig_multiturn(items: pd.DataFrame, out_dir: str):
    """P(correct) across rounds (0 = initial answer) for the multi-turn subset."""
    dss, models = _datasets(items), _models(items)
    arms = [("mt_escalating_gold", "Escalating → correct option", BLUE, "-", "o"),
            ("mt_escalating_wrong", "Escalating → wrong option", ORANGE, "-", "o"),
            ("mt_constant_wrong", "Constant → wrong option", ORANGE, (0, (3, 2)), "s")]
    panels = [(ds, grp) for ds in dss for grp in ("correct", "incorrect")]
    fig, axes = plt.subplots(len(panels), len(models), figsize=(2.5 * len(models), 2.1 * len(panels)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (ds, grp) in enumerate(panels):
        for j, m in enumerate(models):
            ax = axes[i][j]
            g = items[(items.model_name == m) & (items.dataset == ds) & (items.initial_status == grp)]
            g = g[g["mt_escalating_wrong_r1_correct"].notna()] if "mt_escalating_wrong_r1_correct" in g \
                else g.iloc[0:0]
            n = len(g)
            for arm, label, col, ls, mk in (arms if n else []):
                ys, lo, hi = [1.0 if grp == "correct" else 0.0], [], []
                for r in (1, 2, 3):
                    s = g[f"{arm}_r{r}_correct"].astype(bool) if n else pd.Series(dtype=bool)
                    p, l, h = wilson(int(s.sum()), n) if n else (np.nan, np.nan, np.nan)
                    ys.append(p)
                    lo.append(l)
                    hi.append(h)
                ax.plot(range(4), ys, color=col, ls=ls, marker=mk, ms=4, lw=1.6, label=label if (i, j) == (0, 0) else None,
                        zorder=3)
                if n:
                    ax.fill_between(range(1, 4), lo, hi, color=col, alpha=0.10, lw=0, zorder=1)
            ax.set_xticks(range(4), ["init", "R1", "R2", "R3"])
            _pct_axis(ax, 1.03)
            if i == 0:
                ax.set_title(MODEL_LABEL.get(m, m))
            if j == 0:
                ax.set_ylabel(f"{DS_LABEL.get(ds, ds).split(' (')[0]}\ninit. {grp}\nP(correct)", fontsize=8)
            ax.text(1.0, 1.0, f"n={n}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8)
    fig.subplots_adjust(hspace=0.32, wspace=0.12, top=0.93)
    _save(fig, out_dir, "fig5_multiturn_trajectories")


def fig_paraphrase(eff_k: pd.DataFrame, eff: pd.DataFrame, out_dir: str):
    """Forest plot: H1 and H2 effects per paraphrase template vs. pooled, per model x dataset."""
    hyps = [("H1", "H1: false assertion − neutral\nP(still correct), initially correct"),
            ("H2", "H2: discrimination\nP(adopt | true) − P(adopt | false)")]
    dss, models = _datasets(eff), _models(eff)
    rows = [(ds, m) for ds in dss for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.62 * len(rows) + 1.2), sharey=True)
    for ax, (h, title) in zip(axes, hyps):
        for yi, (ds, m) in enumerate(rows):
            y0 = len(rows) - 1 - yi
            pooled = eff[(eff.hypothesis == h) & (eff.model_name == m) & (eff.dataset == ds)]
            if not pooled.empty:
                r = pooled.iloc[0]
                ax.plot([r.ci_lo, r.ci_hi], [y0, y0], color=TEXT, lw=1.6, zorder=2)
                ax.scatter([r["diff"]], [y0], marker="D", s=26, color=TEXT, zorder=3,
                           label="Pooled" if yi == 0 else None)
            for k in sorted(eff_k.paraphrase.dropna().unique()):
                rk = eff_k[(eff_k.hypothesis == h) & (eff_k.model_name == m) & (eff_k.dataset == ds)
                           & (eff_k.paraphrase == k)]
                if rk.empty or rk.iloc[0].n == 0:
                    continue
                rk = rk.iloc[0]
                yk = y0 + 0.3 - 0.15 * k
                ax.plot([rk.ci_lo, rk.ci_hi], [yk, yk], color=MUTED, lw=0.8)
                ax.scatter([rk["diff"]], [yk], s=10, color=MUTED, zorder=3,
                           label="Each paraphrase template" if (yi == 0 and k == 0) else None)
        ax.axvline(0, color=AXIS, lw=0.9)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    axes[0].set_yticks(range(len(rows)), [f"{MODEL_LABEL.get(m, m)} · {DS_LABEL.get(ds, ds).split(' (')[0]}"
                                          for ds, m in reversed(rows)], fontsize=7.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.08), ncol=2, fontsize=7.5)
    _save(fig, out_dir, "fig6_paraphrase_robustness")


def make_all(items, cells, eff, eff_k, breakdown, out_dir):
    _style()
    fig_initial_accuracy(cells, out_dir)
    fig_single_turn(cells, out_dir)
    fig_discrimination(eff, out_dir)
    fig_outcomes(breakdown, out_dir)
    fig_multiturn(items, out_dir)
    fig_paraphrase(eff_k, eff, out_dir)
