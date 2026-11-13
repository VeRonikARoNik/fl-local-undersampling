#!/usr/bin/env python3
"""Figure 1: the experimental pipeline as numbered stages, with the purpose of each stage.
Drawn with explicit coordinates so that no element overlaps another."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
INK, INK2 = "#0b0b0b", "#52514e"
BLUE_BG, BLUE = "#eaf2fc", "#2a78d6"
ORANGE_BG, ORANGE = "#fdeee6", "#eb6834"
GREY_BG = "#f0efec"
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.dpi": 300})

# (number, title, detail lines, purpose lines, band)
STAGES = [
    (1, "Load and clean the data set",
     ["merge source files, drop ID columns,", "remove rows with missing values"],
     ["one clean table per benchmark", "(9 data sets)"], "once"),
    (2, "Stratified 5-fold cross-validation",
     [r"held-out fold $F_i$ → validation 50% + test 50%", r"training part $D_i^{tr} = D \setminus F_i$"],
     ["evaluate on unseen data, five times;", "validation and test are never undersampled"], "loop"),
    (3, r"Split $\mathbf{D_i^{tr}}$ among K clients",
     ["K = 3, 4, ..., 10 disjoint, stratified", r"client sets $D_1, \ldots, D_K$"],
     ["simulate K data holders;", "measure the effect of federation size"], "loop"),
    (4, "Local undersampling on each client",
     ["baseline (none) or one of 9 methods:", "RUS auto/75/50/25%, NearMiss-1/3, ENN, AllKNN, Tomek"],
     ["rebalance and shrink local data;", "the factor under study"], "client"),
    (5, "Local training on each client",
     ["DT, RF, GB or XGBoost,", "fixed hyperparameters, trained once"],
     ["each client builds its own model;", "raw data never leave the client"], "client"),
    (6, "Local prediction",
     [r"class-probability vectors $p_k(x)$", "for validation and test inputs"],
     ["only model outputs are shared", "with the server"], "client"),
    (7, "Aggregation on the server",
     ["6 rules: FedAvg, weighted FedAvg, median,", "trimmed mean, distance-, rank-weighted; argmax"],
     ["combine K clients into one", "federated prediction (one shot)"], "server"),
    (8, "Evaluation on the test set",
     ["F1-macro (primary), accuracy, precision,", "recall, AUC-ROC, local training time"],
     ["one result row per run;", "86 400 runs in total"], "loop"),
    (9, "Statistical analysis",
     ["factorial ANOVA (ω²), Friedman + Nemenyi,", "Wilcoxon signed-rank with Holm correction"],
     ["which factors matter and which", "differences are significant"], "once"),
]

W, H = 7.8, 9.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

x_box, w_box = 0.95, 3.55          # stage boxes
x_pur = x_box + w_box + 0.62       # purpose column
h_box, gap = 0.74, 0.22
y_top = H - 0.75
ys = [y_top - i * (h_box + gap) - h_box for i in range(len(STAGES))]

# header
ax.text(x_box, H - 0.35, "Stage", fontsize=9, fontweight="bold", color=INK, va="center")
ax.text(x_pur, H - 0.35, "Purpose", fontsize=9, fontweight="bold", color=INK, va="center")

# bands behind client-side and server-side stages
def band(i0, i1, fc, ec, label):
    y0 = ys[i1] - 0.09; y1 = ys[i0] + h_box + 0.09
    ax.add_patch(Rectangle((x_box - 0.12, y0), w_box + 0.48, y1 - y0, fc=fc, ec=ec, lw=0.9, zorder=0))
    ax.text(x_box + w_box + 0.18, (y0 + y1) / 2, label, fontsize=6.8, color=ec, ha="center", va="center",
            rotation=90, fontweight="bold")
band(3, 5, BLUE_BG, BLUE, "CLIENT SIDE")
band(6, 6, ORANGE_BG, ORANGE, "SERVER")

# loop bracket for stages 2..8
yb0 = ys[7] + 0.05; yb1 = ys[1] + h_box - 0.05
xb = 0.55
ax.plot([xb + 0.12, xb, xb, xb + 0.12], [yb1, yb1, yb0, yb0], color=INK2, lw=1)
ax.text(xb - 0.12, (yb0 + yb1) / 2, "repeated for every fold i = 1, ..., 5, every K = 3, ..., 10,\nevery method, learner and aggregation rule",
        rotation=90, ha="center", va="center", fontsize=7, color=INK2, linespacing=1.3)

for (num, title, detail, purpose, kind), y in zip(STAGES, ys):
    fc = GREY_BG if kind == "once" else "white"
    ax.add_patch(FancyBboxPatch((x_box, y), w_box, h_box, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=INK2, lw=0.9, zorder=2))
    # number badge
    ax.add_patch(plt.Circle((x_box + 0.27, y + h_box / 2), 0.17, fc=INK, ec="none", zorder=3))
    ax.text(x_box + 0.27, y + h_box / 2, str(num), color="white", fontsize=8.5, fontweight="bold",
            ha="center", va="center", zorder=4)
    tx = x_box + 0.55
    ax.text(tx, y + h_box - 0.17, title, fontsize=8.3, fontweight="bold", color=INK, va="center", zorder=4)
    ax.text(tx, y + h_box - 0.40, detail[0], fontsize=6.9, color=INK2, va="center", zorder=4)
    ax.text(tx, y + h_box - 0.58, detail[1], fontsize=6.9, color=INK2, va="center", zorder=4)
    ax.text(x_pur, y + h_box - 0.27, purpose[0], fontsize=7.4, color=INK, va="center")
    ax.text(x_pur, y + h_box - 0.50, purpose[1], fontsize=7.4, color=INK, va="center")
    # dotted connector to purpose
    ax.plot([x_box + w_box + 0.40, x_pur - 0.06], [y + h_box / 2, y + h_box / 2], color="#c3c2b7", lw=0.6, ls=":")

# arrows between consecutive stages
for y_a, y_b in zip(ys[:-1], ys[1:]):
    ax.annotate("", xy=(x_box + w_box / 2, y_b + h_box + 0.01), xytext=(x_box + w_box / 2, y_a - 0.01),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0), zorder=1)

fig.savefig(OUT / "fig1_pipeline.pdf"); fig.savefig(OUT / "fig1_pipeline.png")
print("ok")
