#!/usr/bin/env python3
"""Figures of the chapter on how the elements of the study affect the results and each other.

Run from analysis/ after interactions.py and paths_compare.py. Each figure is a function; call
`python make_figs_ix.py name1 name2 ...` to draw a subset (default: all)."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats
from figstyle import (INK, INK2, GRID, BLUE, ORANGE, AQUA, YELLOW, RED, NEUTRAL, GREY, LIGHTBLUE, US_LABEL, FL_LABEL,
                      LEARNER_LABEL, DATASETS, METHODS, LEARNERS, MINUS, num, save)

Y = "test_f1_macro"
RULES = ["FedAvg", "WeightedFedAvg", "FedProx", "FedTrimmedMean", "FedBoost", "FedMedian"]
RULE_COL = {"FedAvg": GREY, "WeightedFedAvg": BLUE, "FedProx": AQUA, "FedTrimmedMean": YELLOW, "FedBoost": ORANGE,
            "FedMedian": RED}
LEARNER_COL = {"DT": YELLOW, "RF": AQUA, "GB": ORANGE, "XGB": BLUE}
_df = None


def results():
    global _df
    if _df is None:
        _df = pd.read_csv("fl_results_clean.csv"); _df["F1"] = _df[Y] * 100
    return _df


def kshades(n=8):
    cmap = LinearSegmentedColormap.from_list("k", ["#cfe0f6", "#123f7a"])
    return [cmap(i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------
def fig_ix_matrix():
    """Share of the variance explained by every element (diagonal) and by every pair of elements (below)."""
    a = pd.read_csv("anova_all2way.csv", index_col=0)
    F = [("C(D)", "Data set"), ("C(US)", "Undersampling\nmethod"), ("C(L)", "Learner"), ("C(A)", "FL method"),
         ("C(K)", "Number of\nclients K")]
    n = len(F); M = np.full((n, n), np.nan)
    for i, (fi, _) in enumerate(F):
        for j, (fj, _) in enumerate(F):
            if i == j:
                M[i, j] = a.loc[fi, "omega2"]
            elif i > j:
                for key in (f"{fj}:{fi}", f"{fi}:{fj}"):
                    if key in a.index:
                        M[i, j] = a.loc[key, "omega2"]
    V = np.clip(M, 1e-4, None)
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#123f7a"])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(np.where(np.isnan(M), np.nan, V), cmap=cmap, norm=LogNorm(1e-4, 0.5))
    for i in range(n):
        for j in range(n):
            if np.isnan(M[i, j]):
                continue
            v = max(M[i, j], 0.0)
            s = f"{v:.3f}" if v >= 0.001 else ("< 0.001" if v > 0 else "0.000")
            ax.text(j, i, s, ha="center", va="center", fontsize=8, color="white" if v > 0.05 else INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(n)); ax.set_xticklabels([l for _, l in F], fontsize=7.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([l for _, l in F], fontsize=7.5)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("ω² (share of the total variance; logarithmic colour scale)"); cb.outline.set_visible(False)
    save(fig, "figIxMatrix")


# ---------------------------------------------------------------------------------------------------------------
def fig_ix_main():
    """Mean test F1 per level of every element, all on the same axis that starts at zero."""
    df = results()
    panels = [("dataset", DATASETS, {}, "(a) Data set"),
              ("us_method", ["none"] + METHODS, US_LABEL, "(b) Undersampling"),
              ("local_model", LEARNERS, {"DT": "DT", "RF": "RF", "GB": "GB", "XGB": "XGB"}, "(c) Learner"),
              ("fl_algorithm", RULES, {**FL_LABEL, "FedAvg": "FedAvg"}, "(d) FL method"),
              ("n_clients", list(range(3, 11)), {}, "(e) Clients K")]
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 3.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[9, 10, 4, 6, 8], wspace=0.12))
    for ax, (col, order, lab, title) in zip(axes, panels):
        m = df.groupby(col)["F1"].mean().reindex(order)
        colr = [GREY if (col == "us_method" and k == "none") else BLUE for k in order]
        ax.bar(range(len(order)), m.values, color=colr, width=0.72)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lab.get(k, str(k)) for k in order], rotation=90, fontsize=6.5)
        ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
        rng = m.max() - m.min()
        ax.set_title(f"{title}\nrange {num(rng, 1)}%", loc="left", fontsize=6.8)
    axes[0].set_ylabel("Mean test F1-macro [%]")
    save(fig, "figIxMain")


# ---------------------------------------------------------------------------------------------------------------
def fig_ix_data():
    """How the choice of data set changes the effect of every other element."""
    df = results()
    b = df[df.us_method == "none"]
    base = b.groupby("dataset")["F1"].mean().reindex(DATASETS)
    key = ["dataset", "fold", "n_clients", "local_model", "fl_algorithm"]
    bb = b.set_index(key)["F1"].rename("b")
    u = df[df.us_method != "none"].join(bb, on=key); u["d"] = u.F1 - u.b
    harm = u[u.us_method.isin(["AllKNN", "RUS", "ENN", "NearMiss", "NM3"])].groupby("dataset").d.mean().reindex(DATASETS)
    safe = u[u.us_method.isin(["RUS75", "RUS50"])].groupby("dataset").d.mean().reindex(DATASETS)
    perK = b.groupby(["dataset", "n_clients"])["F1"].mean().unstack()
    kch = (perK[10] - perK[3]).reindex(DATASETS)
    perL = b.groupby(["dataset", "local_model"])["F1"].mean().unstack()
    lsp = (perL.max(axis=1) - perL.min(axis=1)).reindex(DATASETS)
    perA = df.groupby(["dataset", "fl_algorithm"])["F1"].mean().unstack()
    asp = (perA.max(axis=1) - perA.min(axis=1)).reindex(DATASETS)
    props = pd.read_csv("ix_dataset_properties.csv", index_col=0).reindex(DATASETS)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.9), gridspec_kw=dict(hspace=0.75, wspace=0.32))
    spec = [(axes[0, 0], base, "(a) Baseline F1-macro [%]", BLUE, (0, 100)),
            (axes[0, 1], harm, "(b) Change by the five harmful\nmethods [%] (mean)", RED, (-32, 2)),
            (axes[0, 2], safe, "(c) Change by RUS 75% and\nRUS 50% [%] (mean)", AQUA, (-32, 2)),
            (axes[1, 0], kch, "(d) Change of the baseline from\nK = 3 to K = 10 [%]", ORANGE, (-32, 2)),
            (axes[1, 1], lsp, "(e) Best minus worst learner,\nbaseline [%]", YELLOW, (0, 32))]
    for ax, s, title, c, lim in spec:
        ax.bar(range(len(s)), s.values, color=c, width=0.7)
        if c == AQUA:                                  # tiny values: one note instead of crowded labels
            ax.text(0.5, 0.45, f"all data sets between {num(s.min(), 1, sign=True)}%\nand {num(s.max(), 1, sign=True)}%",
                    transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=INK)
        else:
            for x_, v in enumerate(s.values):
                off = (0.6 if v >= 0 else -0.6) * (lim[1] - lim[0]) / 34
                ax.text(x_, v + off, num(v, 1, sign=lim[0] < 0),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=5.6, color=INK)
        ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=90, fontsize=6.5)
        ax.set_ylim(*lim); ax.grid(axis="x", visible=False); ax.set_title(title, loc="left", fontsize=7.5)
        if lim[0] < 0:
            ax.axhline(0, color=INK2, lw=0.8)
    ax = axes[1, 2]
    ax.scatter(props.IR, harm.values, s=26, color=RED, zorder=3)
    offs = {"LR": (-2, 6), "UNSW": (3, -10), "SLS": (5, 2), "SVS": (5, -3)}
    for ds_, x_, y_ in zip(DATASETS, props.IR, harm.values):
        ax.annotate(ds_, (x_, y_), xytext=offs.get(ds_, (4, 2)), textcoords="offset points", fontsize=6.2, color=INK)
    rho = stats.spearmanr(props.IR, harm.values).correlation
    ax.set_xscale("log"); ax.set_xlim(0.8, 1e4); ax.set_ylim(-32, 2); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Imbalance ratio IR (logarithmic)", fontsize=7)
    ax.set_title(f"(f) Panel (b) against the imbalance\nratio (Spearman ρ = {num(rho, 2)})", loc="left", fontsize=7.5)
    save(fig, "figIxData")


# ---------------------------------------------------------------------------------------------------------------
def fig_ix_us():
    """The effect of each undersampling method for every FL method, number of clients and learner."""
    R = pd.read_csv("ix_method_rule.csv", index_col=0).reindex(METHODS)[RULES]
    Kt = pd.read_csv("ix_method_K.csv", index_col=0).reindex(METHODS); Kt.columns = [int(c) for c in Kt.columns]
    L = pd.read_csv("ix_method_learner.csv", index_col=0).reindex(METHODS)[LEARNERS]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    ks = kshades(8)
    groups = [(axes[0], R, [RULE_COL[r] for r in RULES], [FL_LABEL.get(r, r) for r in RULES],
               "(a) per FL method"),
              (axes[1], Kt, ks, [f"K = {k}" for k in Kt.columns], "(b) per number of clients K"),
              (axes[2], L, [LEARNER_COL[l] for l in LEARNERS], [LEARNER_LABEL[l] for l in LEARNERS], "(c) per learner")]
    yy = np.arange(len(METHODS))[::-1]
    for ax, T, cols, labs, title in groups:
        n = T.shape[1]; h = 0.84 / n
        for j in range(n):
            ax.barh(yy + (n / 2 - j - 0.5) * h, T.iloc[:, j].values, height=h * 0.92, color=cols[j], label=labs[j])
        ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
        ax.set_xlim(-32, 3); ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_xlabel("Difference to the baseline [%]", fontsize=7)
        ax.legend(loc="upper left", fontsize=6, handlelength=1.0, labelspacing=0.25, borderaxespad=0.2)
    axes[0].set_yticks(yy); axes[0].set_yticklabels([US_LABEL[m] for m in METHODS])
    save(fig, "figIxUS")


# ---------------------------------------------------------------------------------------------------------------
def fig_ix_rules():
    """The difference of each FL method to FedAvg per data set, number of clients and undersampling condition."""
    order = ["WeightedFedAvg", "FedProx", "FedTrimmedMean", "FedBoost", "FedMedian"]
    D = pd.read_csv("ix_rule_dataset.csv", index_col=0).reindex(order)[DATASETS]
    Kt = pd.read_csv("ix_rule_K.csv", index_col=0).reindex(order)
    M = pd.read_csv("ix_rule_method.csv", index_col=0).reindex(order)[["none"] + METHODS]
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.6), gridspec_kw=dict(hspace=0.75))
    for ax, T, xl, title in [(axes[0], D, list(D.columns), "(a) per data set"),
                             (axes[1], Kt, [f"K = {c}" for c in Kt.columns], "(b) per number of clients K"),
                             (axes[2], M, [US_LABEL[c] for c in M.columns], "(c) per undersampling condition")]:
        im = ax.imshow(T.values, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                v = T.values[i, j]
                ax.text(j, i, num(v, 2, sign=True), ha="center", va="center", fontsize=6.3,
                        color="white" if abs(v) > 1.0 else INK)
        ax.set_xticks(range(T.shape[1])); ax.set_xticklabels(xl, fontsize=6.8, rotation=20, ha="right")
        ax.set_yticks(range(T.shape[0])); ax.set_yticklabels([FL_LABEL[r] for r in T.index], fontsize=6.8)
        ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(title, loc="left", fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Difference to FedAvg [%]"); cb.outline.set_visible(False)
    save(fig, "figIxRules")


# ---------------------------------------------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=6.4, color=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def _arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=7))


def fig_paths():
    """Where undersampling happens relative to the cross-validation split and the client partition."""
    rows = [("Path 1:\nundersampling\nbefore the split\n(US\u2192FL)", "#fbe3dd",
             ["Undersampling\nof the whole\ndata set D", "5-fold split of\nthe under-\nsampled data", "Partition of the\ntraining part\ninto K clients",
              "Local training\nand aggregation", "Evaluation on an\nundersampled\ntest half"], [ORANGE, None, None, None, RED]),
            ("Path 2:\nundersampling\non each client\n(FL\u2192US, this study)", "#e3eefb",
             ["Data set D", "5-fold split:\ntraining part,\nvalidation and\ntest halves", "Partition of the\ntraining part\ninto K clients",
              "Undersampling\non each client,\nlocal training,\naggregation", "Evaluation on\nthe original\ntest half"], [None, None, None, BLUE, None]),
            ("Path 3:\nno undersampling\n(baseline)", NEUTRAL,
             ["Data set D", "5-fold split:\ntraining part,\nvalidation and\ntest halves", "Partition of the\ntraining part\ninto K clients",
              "Local training\nand aggregation", "Evaluation on\nthe original\ntest half"], [None] * 5)]
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.45); ax.axis("off")
    w, h, x0, gap = 1.42, 0.78, 2.45, 0.19
    for r, (name, bg, steps, accents) in enumerate(rows):
        y = 2.45 - r * 1.12
        ax.add_patch(FancyBboxPatch((0.05, y - 0.08), 10.5, h + 0.16, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    fc=bg, ec="none"))
        ax.text(0.15, y + h / 2, name, ha="left", va="center", fontsize=6.3, fontweight="bold", color=INK)
        for k, (s, acc) in enumerate(zip(steps, accents)):
            x = x0 + k * (w + gap)
            _box(ax, x, y, w, h, s, ec=acc or INK2, fc="#fff7f4" if acc in (ORANGE, RED) else "#f3f8fe" if acc else "white",
                 color=INK, fs=5.9)
            if k:
                _arrow(ax, x - gap + 0.01, x - 0.01, y + h / 2)
    save(fig, "figPaths")


def fig_paths_results():
    """Mean test F1 of every method under paths 1 and 2, and the share of the test data that path 1 keeps."""
    P = pd.read_csv("paths_by_method.csv").set_index("method").reindex(METHODS)
    T = pd.read_csv("path1_testsets.csv")
    kept = T[T.part == "test"].groupby("method").kept_pct.mean().reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.08))
    yy = np.arange(len(METHODS))[::-1]; h = 0.38
    ax = axes[0]
    ax.barh(yy + h / 2, P.after_split, height=h * 0.92, color=BLUE, label="Path 2: on each client (correct evaluation)")
    ax.barh(yy - h / 2, P.before_split, height=h * 0.92, color=ORANGE, label="Path 1: before the split (undersampled test data)")
    for y_, a, b_ in zip(yy, P.after_split, P.before_split):
        ax.text(a - 1.0, y_ + h / 2, num(a, 1), va="center", ha="right", fontsize=6.0, color="white")
        ax.text(b_ - 1.0, y_ - h / 2, num(b_, 1), va="center", ha="right", fontsize=6.0, color="white")
    b0 = P.baseline.iloc[0]
    ax.axvline(b0, color=INK, lw=1.0, ls="--", zorder=0.5)
    ax.text(b0 - 1.0, len(METHODS) - 0.35, f"Path 3, baseline: {num(b0, 1)}", ha="right", va="bottom", fontsize=6.5, color=INK)
    ax.set_xlim(0, 105); ax.set_xlabel("Mean test F1-macro [%]"); ax.grid(axis="y", visible=False)
    ax.set_yticks(yy); ax.set_yticklabels([US_LABEL[m] for m in METHODS])
    ax.legend(loc="lower left", fontsize=6.3, bbox_to_anchor=(0.0, -0.36), ncol=1, handlelength=1.2)
    ax.set_title("(a) Mean test F1-macro", loc="left", fontsize=7.5, pad=14)
    ax = axes[1]
    ax.barh(yy, kept.values, height=0.6, color=ORANGE)
    for y_, v in zip(yy, kept.values):
        ax.text(v + 1.0, y_, num(v, 1), va="center", fontsize=6.3, color=INK)
    ax.set_xlim(0, 110); ax.set_xlabel("Test data kept in path 1 [%]"); ax.grid(axis="y", visible=False)
    ax.set_title("(b) Share of the original test half\nthat path 1 evaluates on", loc="left", fontsize=7.5)
    save(fig, "figPathsResults")


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in dir() if n.startswith("fig_")]
    for n in names:
        globals()[n]()
