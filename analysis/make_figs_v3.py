#!/usr/bin/env python3
"""Figures added or redrawn in version 3 of the manuscript (bar charts with axes that start at zero).

Run from analysis/. Inputs: fl_results_clean.csv and the CSV files written by stats_extra.py, nonparam.py,
tables.py, make_table3.py, build_audit_orig.py, data_summary.py and rerun_check.py. Each figure is a function; call
`python make_figs_v3.py name1 name2 ...` to draw a subset (default: all).
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats
from figstyle import (INK, INK2, GRID, BLUE, ORANGE, AQUA, YELLOW, RED, NEUTRAL, GREY, LIGHTBLUE, US_LABEL, FL_LABEL,
                      LEARNER_LABEL, DATASETS, METHODS, LEARNERS, MINUS, num, save)

Y = "test_f1_macro"
_df = None


def results():
    global _df
    if _df is None:
        _df = pd.read_csv("fl_results_clean.csv")
    return _df


def kshades(n=8):
    cmap = LinearSegmentedColormap.from_list("k", ["#cfe0f6", "#123f7a"])
    return [cmap(i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------
def fig_k_bars():
    """Mean F1 of the baseline for every K, per learner and per data set; bars start at zero."""
    df = results(); b = df[df.us_method == "none"]
    per_l = b.groupby(["local_model", "n_clients"])[Y].mean().unstack() * 100
    per_d = b.groupby(["dataset", "n_clients"])[Y].mean().unstack() * 100
    Ks = list(range(3, 11)); col = kshades(len(Ks))
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.0), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.55))
    for ax, tab, order, lab, title in [(axes[0], per_l, LEARNERS, LEARNER_LABEL, "(a) Per learner (mean over the nine data sets)"),
                                       (axes[1], per_d, DATASETS, {}, "(b) Per data set (mean over the four learners)")]:
        w = 0.105
        for g, key in enumerate(order):
            for j, K in enumerate(Ks):
                ax.bar(g + (j - 3.5) * w, tab.loc[key, K], width=w * 0.92, color=col[j], label=f"K = {K}" if g == 0 else None)
            ch = tab.loc[key, 10] - tab.loc[key, 3]
            ax.text(g, 103, f"{num(ch, 1, sign=True)}%", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([lab.get(k, k) for k in order])
        ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Mean test F1-macro [%]")
        ax.grid(axis="x", visible=False); ax.set_title(title, loc="left", pad=14)
        ax.set_xlim(-0.65, len(order) - 0.35)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    save(fig, "figK_bars")


# ---------------------------------------------------------------------------------------------------------------
def fig_cost_bars():
    """Per method: share of local data kept, training time saved, change in F1-macro (three bar panels)."""
    T = pd.read_csv("table_us_methods.csv").set_index("method")
    ret = pd.read_csv("us_audit_per_dataset.csv")
    kept = ret.groupby("method").kept.mean() * 100
    order = METHODS
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.8), sharey=True, gridspec_kw=dict(wspace=0.22))
    yy = np.arange(len(order))[::-1]
    panels = [(axes[0], kept.reindex(order), "(a) Local data\nkept [%]", (0, 110), BLUE),
              (axes[1], T.loc[order, "time_saving"], "(b) Training time\nsaved [%]", (0, 110), AQUA),
              (axes[2], T.loc[order, "delta"], "(c) Change in\nF1-macro [%]", (-30, 6), RED)]
    for ax, vals, title, lim, c in panels:
        ax.barh(yy, vals.values, color=c, height=0.62)
        for y_, v in zip(yy, vals.values):
            off = 1.2 if v >= 0 else -1.0
            txt = num(v, 2 if abs(v) < 0.1 else 1, sign=(lim[0] < 0))
            ax.text(v + (off if lim[1] > 50 else (0.6 if v >= 0 else -0.6)), y_, txt,
                    va="center", ha="left" if v >= 0 else "right", fontsize=6.8, color=INK)
        ax.set_xlim(*lim); ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y", visible=False); ax.axvline(0, color=INK2, lw=0.8)
    axes[0].set_yticks(yy); axes[0].set_yticklabels([US_LABEL[m] for m in order])
    axes[0].set_xticks([0, 50, 100]); axes[1].set_xticks([0, 50, 100])
    axes[2].set_xticks([-20, -10, 0])
    save(fig, "figCost_bars")


# ---------------------------------------------------------------------------------------------------------------
def fig_time_heatmap():
    """Median local training time per federation (baseline, FedAvg) as an annotated table."""
    df = results(); b = df[(df.us_method == "none") & (df.fl_algorithm == "FedAvg")]
    t = b.groupby(["dataset", "local_model"]).train_time_sec.median().unstack().reindex(index=DATASETS, columns=LEARNERS)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#123f7a"])
    im = ax.imshow(t.values, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=300), aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            s = f"{v:.3f}" if v < 0.01 else f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 100 else f"{v:.0f}"
            ax.text(j, i, s, ha="center", va="center", fontsize=7, color="white" if v > 5 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([LEARNER_LABEL[l].replace(" ", "\n") for l in LEARNERS])
    ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("seconds (logarithmic colour scale)"); cb.outline.set_visible(False)
    save(fig, "figTime_heatmap")


# ---------------------------------------------------------------------------------------------------------------
def fig_prec_recall():
    """Change in macro precision, macro recall and F1-macro per method (bars from zero)."""
    df = results()
    keys = ["dataset", "fold", "n_clients", "fl_algorithm", "local_model"]
    rows = {}
    for mname, col in [("Precision (macro)", "test_precision_macro"), ("Recall (macro)", "test_recall_macro"), ("F1-macro", Y)]:
        p = df.pivot_table(index=keys, columns="us_method", values=col, aggfunc="first")
        d = p.drop(columns="none").sub(p["none"], axis=0) * 100
        rows[mname] = d.groupby(level="dataset").mean().mean()
    D = pd.DataFrame(rows).reindex(METHODS)
    D.to_csv("delta_prec_recall_f1.csv")
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    cols = {"Precision (macro)": ORANGE, "Recall (macro)": AQUA, "F1-macro": BLUE}
    h = 0.26; yy = np.arange(len(METHODS))[::-1]
    for j, (k, c) in enumerate(cols.items()):
        ys = yy + (1 - j) * h
        ax.barh(ys, D[k].values, height=h * 0.9, color=c, label=k)
        for y_, v in zip(ys, D[k].values):
            ax.text(v + (0.3 if v >= 0 else -0.3), y_, num(v, 2 if abs(v) < 0.1 else 1, sign=True), va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.3, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([US_LABEL[m] for m in METHODS])
    ax.set_xlim(-24, 3); ax.set_xlabel("Mean difference to the baseline [%]")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0))
    save(fig, "figPrecRecall")


# ---------------------------------------------------------------------------------------------------------------
def fig_delta_ci():
    """Mean paired differences with 95% cluster-bootstrap CIs on a common scale: methods and rules."""
    T = pd.read_csv("table_us_methods.csv").set_index("method").drop(index="none")
    R = pd.read_csv("rules_vs_fedavg.csv").set_index("rule")
    T = T.reindex(METHODS); R = R.reindex(["WeightedFedAvg", "FedProx", "FedTrimmedMean", "FedBoost", "FedMedian"])
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[len(T), len(R)], hspace=0.35))
    for ax, tab, lab, title in [(axes[0], T.rename(columns={"delta": "d"}), US_LABEL, "(a) Undersampling methods versus the baseline"),
                                (axes[1], R.rename(columns={"mean_delta": "d"}), FL_LABEL, "(b) Aggregation rules versus FedAvg")]:
        yy = np.arange(len(tab))[::-1]
        for y_, (k, r) in zip(yy, tab.iterrows()):
            sig = r.ci_hi < 0 or r.ci_lo > 0
            c = RED if sig and r.d < 0 else BLUE if sig else GREY
            ax.barh(y_, r.d, height=0.6, color=c)
            ax.plot([r.ci_lo, r.ci_hi], [y_, y_], color=INK, lw=1.0)
            for x_ in (r.ci_lo, r.ci_hi):
                ax.plot([x_, x_], [y_ - 0.15, y_ + 0.15], color=INK, lw=1.0)
            x_text = min(r.ci_lo, r.d) - 0.5
            ci = lambda v: num(v, 3) if 0 < abs(v) < 0.005 else num(v, 2)
            ax.text(x_text, y_, f"{num(r.d, 2, sign=True)}  [{ci(r.ci_lo)}, {ci(r.ci_hi)}]",
                    va="center", ha="right", fontsize=6.5, color=INK)
        ax.set_yticks(yy); ax.set_yticklabels([lab[k] for k in tab.index])
        ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    axes[1].set_xlim(-45, 3); axes[1].set_xlabel("Mean paired difference in test F1-macro [%] with 95% CI")
    from matplotlib.patches import Patch
    both = pd.concat([T[["ci_lo", "ci_hi"]], R[["ci_lo", "ci_hi"]]])
    handles = [Patch(color=RED, label="worse: CI below zero"), Patch(color=GREY, label="CI includes zero")]
    if (both.ci_lo > 0).any():
        handles.insert(0, Patch(color=BLUE, label="better: CI above zero"))
    axes[1].legend(handles=handles, loc="center left", ncol=1)
    save(fig, "figDeltaCI")


# ---------------------------------------------------------------------------------------------------------------
def fig_omega2():
    """omega squared of every ANOVA term: F1 as measured and F1 rank-transformed."""
    c = pd.read_csv("anova_raw_vs_rank.csv", index_col=0)
    NAME = {"C(dataset)": "Data set", "C(US)": "Undersampling method", "C(US):C(dataset)": "Method × data set",
            "C(local_model):C(dataset)": "Learner × data set", "C(dataset):C(K)": "Data set × K",
            "C(US):C(K)": "Method × K", "C(K)": "K (number of clients)", "C(local_model)": "Learner",
            "C(US):C(local_model)": "Method × learner", "C(fl_algorithm)": "Aggregation rule",
            "C(US):C(fl_algorithm)": "Method × rule"}
    c = c.sort_values("omega2_raw", ascending=False)
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    yy = np.arange(len(c))[::-1]; h = 0.36
    for j, (col, colr, lab) in enumerate([("omega2_raw", BLUE, "F1-macro (as measured)"),
                                          ("omega2_rank", ORANGE, "F1-macro rank-transformed")]):
        v = c[col].clip(lower=0).values
        ax.barh(yy + (0.5 - j) * h, v, height=h * 0.9, color=colr, label=lab)
        for y_, x_ in zip(yy + (0.5 - j) * h, v):
            ax.text(x_ + 0.005, y_, f"{x_:.4f}" if x_ < 0.01 else f"{x_:.3f}", va="center", fontsize=6.3, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([NAME[i] for i in c.index])
    ax.set_xlim(0, 0.6); ax.set_xlabel("ω² (share of the total variance explained by the term)")
    ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
    save(fig, "figOmega2")


# ---------------------------------------------------------------------------------------------------------------
def fig_residuals():
    """Residual diagnostics of the factorial ANOVA: histogram, normal Q-Q plot, spread per method."""
    z = np.load("anova_residuals.npz"); r = z["resid"].astype(float) * 100
    sd = pd.read_csv("resid_sd_by_method.csv", index_col=0).iloc[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1, 1.1]))
    ax = axes[0]
    bins = np.arange(-40, 41, 2)
    ax.hist(np.clip(r, -40, 40), bins=bins, color=BLUE, density=True)
    xs = np.linspace(-40, 40, 400); ax.plot(xs, stats.norm.pdf(xs, 0, r.std()), color=INK, lw=1)
    ax.set_xlabel("Residual [%]"); ax.set_ylabel("Density"); ax.set_title("(a) Distribution", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    rng = np.random.default_rng(1); s = np.sort(rng.choice(r, 4000, replace=False))
    q = stats.norm.ppf((np.arange(1, len(s) + 1) - 0.5) / len(s)) * r.std()
    ax.plot(q, s, "o", ms=1.6, color=BLUE, alpha=0.6, mec="none")
    lim = 45; ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Normal quantile [%]"); ax.set_ylabel("Residual quantile [%]"); ax.set_title("(b) Normal Q-Q plot", loc="left")
    ax = axes[2]
    order = ["none"] + METHODS
    v = sd.reindex(order).values; yy = np.arange(len(order))[::-1]
    ax.barh(yy, v, color=[GREY] + [BLUE] * len(METHODS), height=0.65)
    for y_, x_ in zip(yy, v):
        ax.text(x_ + 0.2, y_, f"{x_:.1f}", va="center", fontsize=6.3)
    ax.set_yticks(yy); ax.set_yticklabels([US_LABEL[m] for m in order], fontsize=6.8)
    ax.set_xlim(0, max(v) * 1.25); ax.set_xlabel("Residual SD [%]"); ax.set_title("(c) Spread per method", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "figResiduals")


# ---------------------------------------------------------------------------------------------------------------
def fig_landscape():
    """Stage 1: the nine data sets by size, imbalance ratio and size of the smallest class."""
    s = pd.read_csv("data_summary.csv").set_index("code")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw=dict(wspace=0.32))
    offs = [{"UNSW": (0, -10, "center"), "RT": (7, 0, "left"), "LR": (8, 9, "left"), "DRY": (7, 6, "left"),
             "CM": (7, 0, "left"), "SLS": (-6, 7, "right"), "CAR": (7, 0, "left"), "SVS": (7, 0, "left"), "AIH": (0, 9, "center")},
            {"UNSW": (-7, 0, "right"), "RT": (7, 0, "left"), "LR": (9, 4, "left"), "DRY": (8, -7, "left"),
             "CM": (7, 0, "left"), "SLS": (-7, 0, "right"), "CAR": (7, 0, "left"), "SVS": (7, 0, "left"), "AIH": (7, 0, "left")}]
    for off, (ax, col, ylab, title) in zip(offs, [(axes[0], "IR", "Imbalance ratio IR (logarithmic)", "(a) Size and imbalance"),
                                 (axes[1], "smallest", "Instances of the smallest class (logarithmic)", "(b) Size and smallest class")]):
        for code, r in s.iterrows():
            c = RED if r.IR > 6 else BLUE
            ax.scatter(r.instances, r[col], s=18 + 5 * r.classes, color=c, edgecolor="white", lw=0.8, zorder=3)
            dx, dy, ha = off[code]
            ax.annotate(f"{code} ({r.classes})", (r.instances, r[col]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.8, ha=ha, va="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(150, 6e5); ax.set_xlabel("Instances (logarithmic)"); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
    axes[0].axhline(2.5, color=INK2, lw=0.7, ls="--"); axes[0].axhline(6, color=INK2, lw=0.7, ls=":")
    axes[0].text(170, 2.75, "IR = 2.5", fontsize=6.4, color=INK2, va="bottom", ha="left")
    axes[0].text(5.8e5, 6.6, "IR = 6", fontsize=6.4, color=INK2, va="bottom", ha="right")
    axes[0].set_ylim(0.6, 1e4); axes[1].set_ylim(8, 3e5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=RED, label="imbalanced (IR > 6)"),
         Line2D([], [], marker="o", ls="", color=BLUE, label="close to balanced (IR < 2.5)")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    save(fig, "figLandscape")


# ---------------------------------------------------------------------------------------------------------------
def fig_partition():
    """Stages 2 and 3 on Car Evaluation (fold 1): class composition of every set, drawn to scale."""
    p = pd.read_csv("partition_car_f1.csv")
    classes = ["unacc", "acc", "good", "vgood"]; colc = dict(zip(classes, [BLUE, ORANGE, AQUA, YELLOW]))
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    yy = np.arange(len(p))[::-1]
    for y_, (_, r) in zip(yy, p.iterrows()):
        left = 0
        for c in classes:
            ax.barh(y_, r[c], left=left, color=colc[c], height=0.62, edgecolor="white", lw=0.6,
                    label=c if y_ == yy[0] else None)
            left += r[c]
        ax.text(left + 15, y_, f"{int(left)}  ({' / '.join(str(int(r[c])) for c in classes)})", va="center", fontsize=6.8)
    names = {"Data set D": "Stage 1: data set D", "Training part": "Stage 2: training part of fold 1",
             "Validation half": "Stage 2: validation half (never undersampled)", "Test half": "Stage 2: test half (never undersampled)"}
    ax.set_yticks(yy); ax.set_yticklabels([names.get(l, "Stage 3: " + l.lower()) for l in p.label])
    ax.set_xlim(0, 2150); ax.set_xlabel("Instances"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=4, title="class", title_fontsize=7)
    save(fig, "figPartition")


# ---------------------------------------------------------------------------------------------------------------
def fig_clients_k():
    """Stage 3: median client size and median size of the smallest class on a client, per data set and K."""
    a = pd.read_csv("audit_sizes_orig.csv").drop_duplicates(["dataset", "fold", "K", "client"])
    size = a.groupby(["dataset", "K"]).n_before.median().unstack().reindex(DATASETS)
    small = a.groupby(["dataset", "K"]).min_class.median().unstack().reindex(DATASETS)
    lt3 = a.assign(f=a.min_class < 3).groupby(["dataset", "K"]).f.mean().unstack().reindex(DATASETS) * 100
    size.to_csv("client_size_median.csv"); small.to_csv("client_smallest_median.csv"); lt3.to_csv("client_smallest_lt3_share.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1), gridspec_kw=dict(wspace=0.18))
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#123f7a"])
    def kfmt(v):
        return f"{v / 1000:.1f}k" if v >= 10000 else f"{v:,.0f}".replace(",", "\u2009")
    for ax, tab, title, fmtf in [(axes[0], size, "(a) Instances on a client (median)", kfmt),
                                 (axes[1], small, "(b) Instances of the smallest class on a client (median)", kfmt)]:
        im = ax.imshow(np.log10(tab.values.astype(float)), cmap=cmap, aspect="auto", vmin=0, vmax=5)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.values[i, j]
                bold = (tab is small) and v < 3
                ax.text(j, i, fmtf(v), ha="center", va="center", fontsize=6.4,
                        color="white" if np.log10(max(v, 1)) > 3 else (RED if bold else INK), fontweight="bold" if bold else None)
        ax.set_xticks(range(8)); ax.set_xticklabels([f"K={k}" for k in range(3, 11)], fontsize=6.6)
        ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
        ax.grid(False); ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_title(title, loc="left", fontsize=7.6)
    axes[1].set_yticklabels([])
    save(fig, "figClientsK")


# ---------------------------------------------------------------------------------------------------------------
def _cd_panel(ax, ranks, cd, labels, title):
    ranks = ranks.sort_values(); k = len(ranks); vals = ranks.values
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques): cliques.append((i, j))
    half = int(np.ceil(k / 2)); step = 0.42
    y_cl = [-0.32 - 0.22 * n for n in range(len(cliques))]
    y0 = (y_cl[-1] if y_cl else -0.1) - 0.38
    y_min = y0 - step * (half - 1) - 0.25
    ax.set_xlim(1, k); ax.set_ylim(y_min, 1.75); ax.axis("off")
    ax.plot([1, k], [0, 0], color=INK, lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.1], color=INK, lw=0.8); ax.text(r, 0.16, str(r), ha="center", va="bottom", fontsize=7)
    ax.plot([1, 1 + cd], [0.85, 0.85], color=INK, lw=1.2, solid_capstyle="butt")
    for xx in (1, 1 + cd): ax.plot([xx, xx], [0.79, 0.91], color=INK, lw=1.0)
    ax.text(1 + cd / 2, 0.98, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    for i, (m, r) in enumerate(ranks.items()):
        left = i < half
        yy = y0 - step * (i if left else k - 1 - i)
        xe = 1 - 0.12 * (k - 1) / 9 if left else k + 0.12 * (k - 1) / 9
        ax.plot([r, r], [0, yy], color=INK2, lw=0.7, zorder=1); ax.plot([r, xe], [yy, yy], color=INK2, lw=0.7, zorder=1)
        ax.text(xe + (-0.05 if left else 0.05) * (k - 1) / 9, yy, f"{labels.get(str(m), str(m))} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=7, clip_on=False)
    for (a_, b_), yy in zip(cliques, y_cl):
        ax.plot([vals[a_] - 0.04, vals[b_] + 0.04], [yy, yy], color=INK, lw=2.4, solid_capstyle="round", zorder=3)
    ax.text(1, 1.75, title, ha="left", va="top", fontsize=8, fontweight="bold", transform=ax.transData)
    return 1.75 - y_min


def fig_cd_all():
    fr = pd.read_csv("friedman_summary.csv")
    spec = [("undersampling method", US_LABEL, "(a) Undersampling methods (k = 10)"),
            ("learner (baseline)", LEARNER_LABEL, "(b) Learners, baseline (k = 4)"),
            ("aggregation rule", FL_LABEL, "(c) Aggregation rules (k = 6)"),
            ("number of clients K (baseline)", {str(k): f"K = {k}" for k in range(3, 11)}, "(d) Number of clients K, baseline (k = 8)")]
    hs = []
    for name, lab, _ in spec:
        k = int(fr.set_index("factor").loc[name, "k"]); half = int(np.ceil(k / 2))
        hs.append(1.75 + 0.32 + 0.22 * 3 + 0.38 + 0.42 * (half - 1) + 0.25)
    fig, axes = plt.subplots(4, 1, figsize=(5.6, 0.36 * sum(hs) + 0.6), gridspec_kw=dict(height_ratios=hs, hspace=0.25))
    for ax, (name, lab, title) in zip(axes, spec):
        row = fr.set_index("factor").loc[name]
        ranks = pd.Series(json.loads(row.ranks)); ranks.index = ranks.index.astype(str)
        _cd_panel(ax, ranks, row.CD, lab, title)
    save(fig, "figCD_all")


# ---------------------------------------------------------------------------------------------------------------
def class_retention_table():
    """Per client set and method: share kept of the largest class, the intermediate classes and the smallest class."""
    a = pd.read_csv("audit_classes_orig.csv", dtype={"cls": str})
    key = ["dataset", "fold", "K", "client", "method"]
    # largest = first, smallest = last; among classes of equal size, the one the method reduced is taken as the
    # largest and the one it kept whole as the smallest (the class imbalanced-learn treats as the minority)
    a["kept_whole"] = a.n_after == a.n_before
    a = a.sort_values(key + ["n_before", "kept_whole", "cls"], ascending=[True] * 5 + [False, True, True])
    a["pos"] = a.groupby(key).cumcount()
    a["ncls"] = a.groupby(key).cls.transform("size")
    a["group"] = np.where(a.pos == 0, "largest", np.where(a.pos == a.ncls - 1, "smallest", "intermediate"))
    g = a.groupby(key + ["group"]).agg(nb=("n_before", "sum"), na=("n_after", "sum")).reset_index()
    g["kept"] = g.na / g.nb * 100
    per_set = g.pivot_table(index=key, columns="group", values="kept").reset_index()
    a["lost90"] = (a.n_after < 0.1 * a.n_before) & (a.group != "largest")
    a["gone"] = a.n_after == 0
    flags = a.groupby(key).agg(lost90=("lost90", "any"), gone=("gone", "any"), fallback=("fallback", "first")).reset_index()
    per_set = per_set.merge(flags, on=key)
    per_set.to_csv("class_retention_per_clientset.csv", index=False)
    ds = per_set.groupby(["method", "dataset"])[["largest", "intermediate", "smallest", "lost90", "gone"]].mean()
    ds.to_csv("class_retention_per_dataset.csv")
    meth = ds.groupby("method").mean()
    meth[["lost90", "gone"]] *= 100
    meth.to_csv("class_retention_per_method.csv")
    return meth, ds


def fig_class_retention():
    meth, ds = class_retention_table()
    meth = meth.reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.08), sharey=True)
    ax = axes[0]; yy = np.arange(len(METHODS))[::-1]; h = 0.26
    for j, (col, c, lab) in enumerate([("largest", BLUE, "largest class"), ("intermediate", AQUA, "intermediate classes"),
                                       ("smallest", YELLOW, "smallest class")]):
        v = meth[col].values
        ax.barh(yy + (1 - j) * h, v, height=h * 0.9, color=c, label=lab)
    ax.set_xlim(0, 100); ax.set_xlabel("Instances kept [%]"); ax.grid(axis="y", visible=False)
    ax.set_yticks(yy); ax.set_yticklabels([US_LABEL[m] for m in METHODS])
    ax.set_title("(a) Share kept, by class size on the client", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, handlelength=1.2, columnspacing=1.0)
    ax = axes[1]
    v = meth["gone"].values
    ax.barh(yy, v, height=0.6, color=RED)
    for y_, x_ in zip(yy, v):
        ax.text(x_ + 1.5, y_, num(x_, 1), va="center", fontsize=6.8)
    ax.set_xlim(0, 100); ax.set_xlabel("Client sets [%]"); ax.grid(axis="y", visible=False)
    ax.set_title("(b) A class removed entirely", loc="left")
    save(fig, "figClassRetention")


# ---------------------------------------------------------------------------------------------------------------
def local_fed_table():
    """Per data set and condition: F1 of the local models on their own (per client, mean over the four learners)
    and of the federated FedAvg model (mean over the four learners); re-run of fold 1 with K = 5."""
    # outputs of rerun_check.py (about two hours); without a new re-run, the copies stored in ../results/rerun/ are used
    import os
    rr = "rerun" if os.path.exists("rerun/rerun_runs_f1_K5.csv") else "../results/rerun"
    c = pd.read_csv(f"{rr}/rerun_clients_f1_K5.csv"); r = pd.read_csv(f"{rr}/rerun_runs_f1_K5.csv")
    # federations in which a class vanished from every client are scored over all classes (rescore_vanished.py)
    vc = pd.read_csv("rescored_vanished_clients.csv"); vr = pd.read_csv("rescored_vanished_runs.csv")
    kc = ["dataset", "fold", "K", "us_method", "local_model", "client"]
    c = c.merge(vc[kc + ["f1_local_orig", "f1_local_corr"]], on=kc, how="left")
    hit = c.f1_local_corr.notna()
    assert np.allclose(c.loc[hit, "f1_local"], c.loc[hit, "f1_local_orig"])      # same models as in the re-run
    c.loc[hit, "f1_local"] = c.loc[hit, "f1_local_corr"]
    kr = ["dataset", "fold", "K", "us_method", "fl_algorithm", "local_model"]
    r = r.merge(vr[kr + ["test_f1_macro_corr"]], on=kr, how="left")
    r["test_f1_macro_rerun"] = r.test_f1_macro_corr.fillna(r.test_f1_macro_rerun)
    loc = c.groupby(["dataset", "us_method", "client"]).f1_local.mean().mul(100).reset_index()
    fed = r[r.fl_algorithm == "FedAvg"].groupby(["dataset", "us_method"]).test_f1_macro_rerun.mean().mul(100).rename("fed")
    t = loc.groupby(["dataset", "us_method"]).f1_local.agg(local_mean="mean", local_min="min", local_max="max").join(fed)
    t["gain_vs_mean"] = t.fed - t.local_mean; t["gain_vs_best"] = t.fed - t.local_max
    t.to_csv("local_vs_federated_f1_K5.csv")
    return t, loc


def fig_local_fed(methods=("none", "RUS75", "ENN")):
    """Stages 6 and 7: local models on their own against the federated model (re-run of fold 1, K = 5)."""
    t, loc = local_fed_table()
    fig, axes = plt.subplots(len(methods), 1, figsize=(6.3, 1.75 * len(methods) + 0.5), sharex=True,
                             gridspec_kw=dict(hspace=0.45))
    for ax, m in zip(axes, methods):
        first = True
        for i, d in enumerate(DATASETS):
            if (d, m) not in t.index:
                continue
            row = t.loc[(d, m)]; v = loc[(loc.dataset == d) & (loc.us_method == m)].f1_local.values
            ax.bar(i - 0.18, row.local_mean, width=0.34, color=LIGHTBLUE,
                   label="local models on their own (mean; dots: clients)" if first else None)
            ax.plot([i - 0.18, i - 0.18], [v.min(), v.max()], color=INK2, lw=0.9)
            ax.scatter(np.full(len(v), i - 0.18), v, s=5, color=INK2, zorder=3)
            ax.bar(i + 0.18, row.fed, width=0.34, color=BLUE, label="federated model (FedAvg)" if first else None)
            first = False
            ax.text(i + 0.18, row.fed + 2, f"{row.fed:.0f}", ha="center", va="bottom", fontsize=6.3)
        ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="x", visible=False)
        ax.set_title(f"({'abc'[list(methods).index(m)]}) {US_LABEL[m]}", loc="left")
        ax.set_ylabel("Test F1-macro [%]")
    axes[-1].set_xticks(range(len(DATASETS))); axes[-1].set_xticklabels(DATASETS)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    save(fig, "figLocalFed")


# ---------------------------------------------------------------------------------------------------------------
def fig_boxplots():
    """Distribution of test F1-macro per condition (axis from zero)."""
    df = results(); order = ["none"] + METHODS
    data = [df[df.us_method == m][Y].values * 100 for m in order]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.3), whiskerprops=dict(color=INK2, lw=0.9),
                    capprops=dict(color=INK2, lw=0.9), boxprops=dict(lw=0.9, color=INK2))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(GREY if i == 0 else BLUE); b.set_alpha(0.95)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([US_LABEL[m] for m in order], rotation=20, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("Test F1-macro [%]"); ax.grid(axis="x", visible=False)
    save(fig, "figC_f1_boxplots")


def fig_class_distributions():
    """Class shares of every data set (axis from zero), with readable class names."""
    c = pd.read_csv("class_counts.csv"); s = pd.read_csv("data_summary.csv").set_index("code")
    NAMES = {"UNSW": {"0": "normal (0)", "1": "attack (1)"}, "AIH": {"n.d.": "no data (n.d.)"},
             "SLS": {"1": "red soil", "2": "cotton crop", "3": "grey soil", "4": "damp grey soil",
                     "5": "vegetation stubble", "7": "very damp grey soil"}}
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.9))
    th = lambda v: f"{v:,}".replace(",", " ")
    for ax, code in zip(axes.flat, DATASETS):
        cc = c[c.code == code].copy(); cc["cls"] = cc.cls.astype(str); cc = cc.sort_values("n")
        ax.barh(range(len(cc)), cc.share_pct.values, color=BLUE, height=0.7)
        ax.grid(axis="y", visible=False); ax.set_xlim(0, cc.share_pct.max() * 1.18)
        import matplotlib.ticker as mt
        ax.xaxis.set_major_locator(mt.MaxNLocator(4))
        if len(cc) <= 12:
            ax.set_yticks(range(len(cc))); ax.set_yticklabels([NAMES.get(code, {}).get(x, x)[:26] for x in cc.cls], fontsize=6.3)
        else:
            ax.set_yticks([]); ax.set_ylabel(f"{len(cc)} classes", fontsize=7)
        r = s.loc[code]; ir = r.IR
        irs = th(round(ir)) if ir >= 100 else f"{ir:.1f}"
        ax.set_title(f"{code}: n = {th(int(r.instances))}\nIR = {irs}, smallest class = {th(int(r.smallest))}",
                     loc="left", fontsize=7.6, fontweight="normal", linespacing=1.25)
    for ax in axes[-1]:
        ax.set_xlabel("Share of instances [%]")
    fig.tight_layout()
    save(fig, "figA_class_distributions")


# ---------------------------------------------------------------------------------------------------------------
def fig_heatmap():
    """Mean difference to the baseline per data set and method (diverging scale, grey midpoint)."""
    df = results()
    keys = ["dataset", "fold", "n_clients", "fl_algorithm", "local_model"]
    piv = df.pivot_table(index=keys, columns="us_method", values=Y, aggfunc="first")
    delta = piv.drop(columns="none").sub(piv["none"], axis=0) * 100
    H = delta.groupby(level="dataset").mean().reindex(index=DATASETS, columns=METHODS)
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    im = ax.imshow(H.values, cmap=cmap, vmin=-45, vmax=45, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            ax.text(j, i, num(v, 1, sign=True), ha="center", va="center", fontsize=7, color="white" if abs(v) > 28 else INK)
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels([US_LABEL[m] for m in H.columns], rotation=25, ha="right")
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Δ F1-macro [%]"); cb.outline.set_visible(False)
    save(fig, "fig3_delta_heatmap")


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in dir() if n.startswith("fig_")]
    for n in names:
        globals()[n if n.startswith("fig_") else "fig_" + n]()
