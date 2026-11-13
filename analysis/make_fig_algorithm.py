#!/usr/bin/env python3
"""Flowchart of Algorithm 1 (nested loops of the experiment) and of the statistical analysis (Algorithm 2)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import INK, INK2, BLUE, ORANGE, save

CLIENT_FILL, SERVER_FILL, HEAD_FILL = "#e8f0fb", "#fdeee6", "#f0efec"


def box(ax, x0, y_top, w, h, text, fill="white", edge=INK2, size=7.2, bold_first=False, align="center"):
    ax.add_patch(FancyBboxPatch((x0, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec=edge, lw=0.9, zorder=3))
    xt = x0 + w / 2 if align == "center" else x0 + 0.12
    ax.text(xt, y_top - h / 2, text, ha=align, va="center", fontsize=size, color=INK, zorder=4, linespacing=1.35)
    return y_top - h


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2, zorder=2))


def frame(ax, x0, x1, y_top, y_bot, header, count, note=None):
    ax.add_patch(FancyBboxPatch((x0, y_bot), x1 - x0, y_top - y_bot, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="none", ec=INK2, lw=0.9, zorder=1))
    ax.add_patch(FancyBboxPatch((x0, y_top - 0.30), x1 - x0, 0.30, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=HEAD_FILL, ec=INK2, lw=0.9, zorder=1))
    ax.text(x0 + 0.1, y_top - 0.15, header, ha="left", va="center", fontsize=7.4, fontweight="bold", color=INK, zorder=4)
    ax.text(x1 - 0.1, y_top - 0.15, f"\u21bb {count}", ha="right", va="center", fontsize=7.4, color=INK, zorder=4)
    if note:
        ax.text(x1 - 0.1, y_bot + 0.07, note, ha="right", va="bottom", fontsize=6.4, style="italic", color=INK2, zorder=4)


def algorithm1():
    W = 6.3
    fig = plt.figure(figsize=(W, 8.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, 8.0); ax.axis("off")
    cx = W / 2
    y = 7.9
    y = box(ax, 0.45, y, W - 0.9, 0.42, "Input: data set D; set M of nine undersampling methods; learners L = {DT, RF, GB, XGB};\n"
                                        "set R of six aggregation rules; fixed random seeds")
    arrow(ax, cx, y, y - 0.2); y -= 0.2
    y = box(ax, 0.9, y, W - 1.8, 0.34, "Stage 1: prepare D (read and merge files, remove identifier and leaking columns)")
    arrow(ax, cx, y, y - 0.22); y -= 0.22
    # nested loop frames: (header, count, note)
    levels = [("for each fold i = 1, ..., 5   (stage 2)", "5 folds", None),
              ("for each number of clients K = 3, ..., 10   (stage 3)", "8 sizes", None),
              ("for each condition u in {none} \u222a M   (stage 4)", "10 conditions", "all conditions share the client partition"),
              ("for each learner h in L   (stages 5 and 6)", "4 learners", None),
              ("for each rule a in R   (stages 7 and 8)", "6 rules", "all rules reuse the same local models")]
    contents = [
        ("Split D into the held-out fold $F_i$ and the training part $D_i^{tr}$; split $F_i$ into a validation half\n"
         "and a test half (stratified). Validation and test data are never undersampled.", "white", INK2, 0.46),
        ("Partition $D_i^{tr}$ into K disjoint, stratified client sets $D_1, ..., D_K$", "white", INK2, 0.32),
        ("Client side: every client k computes $D_k^{*} = u(D_k)$ on its own data;\n"
         "if u cannot be applied, the client keeps $D_k^{*} = D_k$", CLIENT_FILL, BLUE, 0.46),
        ("Client side: every client k whose $D_k^{*}$ holds at least two classes trains $h_k$ on $D_k^{*}$ once and\n"
         "sends the class-probability vectors $p_k(x)$ for all validation and test inputs x", CLIENT_FILL, BLUE, 0.46),
        ("Server side: $\\bar{p}(x) = a(p_1(x), ..., p_K(x))$,  $\\hat{y}(x) = \\mathrm{argmax}_{c}\\, \\bar{p}_c(x)$;\n"
         "compute the metrics on the test half over all classes of D and store one result row",
         SERVER_FILL, ORANGE, 0.46)]
    x0, x1 = 0.25, W - 0.25
    tops, bottoms = [], []
    for lvl, ((hdr, cnt, note), (txt, fill, edge, h)) in enumerate(zip(levels, contents)):
        tops.append((x0, x1, y))
        yb = y - 0.30 - 0.12
        yb = box(ax, x0 + 0.15, yb, x1 - x0 - 0.30, h, txt, fill=fill, edge=edge, size=7.0)
        if lvl < len(levels) - 1:
            arrow(ax, cx, yb, yb - 0.17); y = yb - 0.17
        else:
            y = yb
        x0 += 0.17; x1 -= 0.17
    # close the frames from the innermost outwards; a frame with a note gets room for it
    yb = y
    for lvl in reversed(range(len(levels))):
        fx0, fx1, ftop = tops[lvl]
        yb -= 0.30 if levels[lvl][2] else 0.12
        frame(ax, fx0, fx1, ftop, yb, levels[lvl][0], levels[lvl][1], levels[lvl][2])
    arrow(ax, cx, yb, yb - 0.22); y = yb - 0.22
    y = box(ax, 0.6, y, W - 1.2, 0.34, "86 400 result rows = 9 data sets \u00d7 5 folds \u00d7 8 sizes \u00d7 10 conditions \u00d7 4 learners \u00d7 6 rules", size=6.9)
    arrow(ax, cx, y, y - 0.2); y -= 0.2
    y = box(ax, 0.9, y, W - 1.8, 0.34, "Stage 9: statistical analysis (Algorithm 2)")
    ax.set_ylim(y - 0.05, 8.0); fig.set_size_inches(W, 8.0 - (y - 0.05))
    save(fig, "figAlgorithm1")


def algorithm2():
    """Workflow of the statistical analysis: three questions, three tools, one decision rule."""
    W, H = 6.3, 5.25
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    y = H - 0.08
    y = box(ax, 1.2, y, W - 2.4, 0.40, "86 400 result rows (test F1-macro of every run)\n"
                                       "factors: data set, method, learner, rule, K; replicates: 5 folds")
    cols = [(0.08, "Q1. How much does each\nfactor matter?", "#e8f0fb", BLUE,
             ["All 86 400 runs", "Factorial ANOVA, type II\nsums of squares: 5 main\neffects, 6 interactions",
              "Effect sizes $\\omega^2$ and\npartial $\\eta^2$ per term", "Residual checks; the same\nmodel on rank-transformed\nF1 (robustness)"]),
            (2.17, "Q2. Do the levels differ when\ndata sets are the units?", "#e6f6ef", "#1baf7a",
             ["Mean F1 per data set and\nlevel: a 9 × k table", "Friedman test with the\nIman-Davenport correction",
              "Nemenyi post hoc test;\ncritical difference (CD)", "CD diagrams: methods,\nlearners, rules, K"]),
            (4.26, "Q3. Is each method different\nfrom the baseline, by how much?", SERVER_FILL, ORANGE,
             ["8 640 matched pairs per\nmethod (same data set, fold,\nK, learner and rule)", "Wilcoxon signed-rank test,\nHolm correction over the\nnine methods",
              "Effect size: matched-pairs\nrank-biserial $r_{rb}$", "Mean difference with 95%\ncluster-bootstrap CI (data\nsets resampled); checked\nwith BCa, t, Bonferroni"])]
    w = 1.96
    y_split = y - 0.18
    ax.plot([W / 2, W / 2], [y, y_split], color=INK2, lw=0.9)
    ax.plot([0.08 + w / 2, 4.26 + w / 2], [y_split, y_split], color=INK2, lw=0.9)
    bottoms = []
    for x0, q, fill, edge, steps in cols:
        arrow(ax, x0 + w / 2, y_split, y_split - 0.14)
        yy = y_split - 0.14
        yy = box(ax, x0, yy, w, 0.46, q, fill=fill, edge=edge, size=7.0)
        ax.texts[-1].set_fontweight("bold")
        for s in steps:
            arrow(ax, x0 + w / 2, yy, yy - 0.15); yy -= 0.15
            h = 0.30 + 0.13 * s.count("\n")
            yy = box(ax, x0, yy, w, h, s, size=6.8)
        bottoms.append(yy)
    yb = min(bottoms) - 0.18
    for (x0, *_), b in zip(cols, bottoms):
        ax.plot([x0 + w / 2, x0 + w / 2], [b, yb], color=INK2, lw=0.9)
    ax.plot([0.08 + w / 2, 4.26 + w / 2], [yb, yb], color=INK2, lw=0.9)
    arrow(ax, W / 2, yb, yb - 0.14)
    y = box(ax, 0.6, yb - 0.14, W - 1.2, 0.42, "A difference counts as established only if it holds with data sets as the units (Nemenyi\n"
                                               "test or a bootstrap CI that excludes zero); $\\omega^2$, the CI and $r_{rb}$ give its size")
    arrow(ax, W / 2, y, y - 0.14)
    y = box(ax, 1.2, y - 0.14, W - 2.4, 0.30, "Answers to the research questions RQ1 to RQ4")
    ax.set_ylim(y - 0.05, H); fig.set_size_inches(W, H - (y - 0.05))
    save(fig, "figAlgorithm2")


if __name__ == "__main__":
    algorithm1()
    algorithm2()
