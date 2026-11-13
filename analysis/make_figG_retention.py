#!/usr/bin/env python3
"""Share of local training data kept by each undersampling method on the clients (audit of the client sets).
Input: us_audit_per_dataset.csv (make_table3.py). Cells marked with * contain clients on which the method could not
be applied and the client kept its data unchanged."""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
OUT = Path("figures"); INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "xtick.color": INK2, "ytick.color": INK2})
DS = ["UNSW", "RT", "LR", "DRY", "CM", "SLS", "CAR", "SVS", "AIH"]
M = ["RUS75", "RUS50", "RUS25", "TomekLinks", "AllKNN", "RUS", "ENN", "NearMiss", "NM3"]
LAB = {"RUS": "RUS (auto)", "RUS25": "RUS 25%", "RUS50": "RUS 50%", "RUS75": "RUS 75%", "NearMiss": "NearMiss-1",
       "NM3": "NearMiss-3", "ENN": "ENN", "AllKNN": "AllKNN", "TomekLinks": "Tomek Links"}
p = pd.read_csv("us_audit_per_dataset.csv")
kept = p.pivot(index="method", columns="dataset", values="kept").reindex(index=M, columns=DS) * 100
fb = p.pivot(index="method", columns="dataset", values="fallback").reindex(index=M, columns=DS) * 100
cmap = LinearSegmentedColormap.from_list("seq", ["#f2f7fd", "#86b6ef", "#1c5cab"])
fig, ax = plt.subplots(figsize=(5.6, 3.3))
im = ax.imshow(kept.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(kept.shape[0]):
    for j in range(kept.shape[1]):
        v = kept.values[i, j]; star = "*" if fb.values[i, j] > 0 else ""
        txt = f"{v:.1f}{star}" if v < 10 else f"{v:.0f}{star}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="white" if v > 62 else INK)
ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS); ax.set_yticks(range(len(M))); ax.set_yticklabels([LAB[m] for m in M])
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Local training data kept [%]"); cb.outline.set_visible(False)
fig.savefig(OUT / "figG_retention.pdf"); fig.savefig(OUT / "figG_retention.png")
print(kept.round(1)); print(fb.round(1))
