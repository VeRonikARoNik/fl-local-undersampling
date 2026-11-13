#!/usr/bin/env python3
"""Additional figures: what each undersampler does to one client (figB, example_client_CAR.csv) and
the baseline F1 per learner and data set (figD). The other figures are drawn by make_figs_v3.py."""
import logging, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})
DS = ["UNSW", "RT", "LR", "DRY", "CM", "SLS", "CAR", "SVS", "AIH"]
FOLD = {"RT": "rt-iot2022", "LR": "letter+recognition", "UNSW": "unsw-nb15", "SLS": "statlog+landsat+satellite",
        "CAR": "car+evaluation", "SVS": "statlog+vehicle+silhouettes", "DRY": "dry+bean+dataset",
        "CM": "crowdsourced+mapping", "AIH": "aviaere_influenza_fallliste"}
US_ORDER = ["none", "RUS75", "RUS50", "RUS25", "TomekLinks", "AllKNN", "RUS", "ENN", "NearMiss", "NM3"]
US_LABEL = {"RUS": "RUS (auto)", "RUS25": "RUS 25%", "RUS50": "RUS 50%", "RUS75": "RUS 75%", "NearMiss": "NearMiss-1",
            "NM3": "NearMiss-3", "ENN": "ENN", "AllKNN": "AllKNN", "TomekLinks": "Tomek Links", "none": "Baseline"}
def save(fig, name): fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)

# ===== What each undersampler does to one client (CAR, fold 1, K = 3, client 1) =====
# class counts of the original client files (orig_audit_all.csv, written by compare_audit_orig.py)
o = pd.read_csv("orig_audit_all.csv", dtype={"cls": str})
o = o[(o.dataset == "CAR") & (o.fold == 1) & (o.K == 3) & (o.client == 1) & ~o.cls.str.startswith("__")]
classes = ["unacc", "acc", "good", "vgood"]; ccol = {"unacc": BLUE, "acc": ORANGE, "good": AQUA, "vgood": YELLOW}
ex = (o.pivot(index="method", columns="cls", values="n").reindex(index=US_ORDER, columns=classes)
      .fillna(0).astype(int))
ex.to_csv("example_client_CAR.csv")
fig, ax = plt.subplots(figsize=(7.0, 3.1))
y = np.arange(len(ex))[::-1]
left = np.zeros(len(ex))
for c in classes:
    ax.barh(y, ex[c].values, left=left, color=ccol[c], height=0.62, edgecolor="white", linewidth=1.2, label=c)
    left += ex[c].values
for yi, (m, r) in zip(y, ex.iterrows()):
    tot = int(r.sum()); parts = " / ".join(str(int(r[c])) for c in classes)
    ax.text(tot + 6, yi, f"{tot}  ({parts})", va="center", fontsize=6.8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([US_LABEL[m] for m in ex.index]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Instances kept on the client"); ax.set_xlim(0, ex.sum(axis=1).max() * 1.32)
ax.legend(title="class", ncol=4, fontsize=7, title_fontsize=7, frameon=False,
          loc="lower right")
save(fig, "figB_example_client_CAR")

# ===== Learner x data set, baseline F1 =====
d = pd.read_csv("fl_results_clean.csv"); Y = "test_f1_macro"
b = d[d.us_method == "none"].groupby(["dataset", "local_model"])[Y].mean().unstack().reindex(index=DS, columns=["DT", "RF", "GB", "XGB"]) * 100
cmap = LinearSegmentedColormap.from_list("seq", ["#f2f7fd", "#86b6ef", "#1c5cab"])
fig, ax = plt.subplots(figsize=(3.6, 3.4))
im = ax.imshow(b.values, cmap=cmap, vmin=55, vmax=100, aspect="auto")
for i in range(b.shape[0]):
    best = b.values[i].argmax()
    for j in range(b.shape[1]):
        v = b.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > 86 else INK,
                fontweight="bold" if j == best else "normal")
ax.set_xticks(range(4)); ax.set_xticklabels(b.columns); ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("Test F1-macro [%]"); cb.outline.set_visible(False)
save(fig, "figD_learner_dataset")
