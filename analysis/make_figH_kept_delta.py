#!/usr/bin/env python3
"""Difference to the baseline (per data set and method) against the share of local data kept on the clients.
Inputs: fl_results_clean.csv (build_clean.py), us_audit_per_dataset.csv (make_table3.py)."""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
OUT = Path("figures"); INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
res = pd.read_csv("fl_results_clean.csv")
f1 = res.groupby(["dataset", "us_method"]).test_f1_macro.mean().unstack() * 100
p = f1.drop(columns="none").sub(f1["none"], axis=0)          # unrounded mean difference per data set
k = pd.read_csv("us_audit_per_dataset.csv"); k["kept"] *= 100
d = p.stack().rename("delta").reset_index().rename(columns={"us_method": "method"})
m = d.merge(k[["method", "dataset", "kept"]], on=["method", "dataset"])
FAM = {"RUS75": "Random, ratio-based (RUS 75/50/25%)", "RUS50": "Random, ratio-based (RUS 75/50/25%)",
       "RUS25": "Random, ratio-based (RUS 75/50/25%)", "RUS": "Random, balancing (RUS auto)",
       "TomekLinks": "Cleaning (Tomek Links, ENN, AllKNN)", "ENN": "Cleaning (Tomek Links, ENN, AllKNN)",
       "AllKNN": "Cleaning (Tomek Links, ENN, AllKNN)", "NearMiss": "Distance-based (NearMiss-1/3)",
       "NM3": "Distance-based (NearMiss-1/3)"}
STY = {"Random, ratio-based (RUS 75/50/25%)": ("#2a78d6", "o"), "Random, balancing (RUS auto)": ("#eda100", "D"),
       "Cleaning (Tomek Links, ENN, AllKNN)": ("#eb6834", "s"), "Distance-based (NearMiss-1/3)": ("#1baf7a", "^")}
m["fam"] = m.method.map(FAM)
m.to_csv("delta_vs_kept.csv", index=False)
rho = spearmanr(m.kept, m.delta).statistic
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for f, (c, mk) in STY.items():
    g = m[m.fam == f]
    ax.scatter(g.kept, g.delta, s=26, color=c, marker=mk, edgecolor="white", linewidth=0.6, label=f, zorder=3)
ax.axhline(0, color=INK2, lw=0.8)
NOTE = {("ENN", "RT"): ("ENN on RT", (84, -17)), ("ENN", "SVS"): ("ENN on SVS", (44, -27)),
        ("RUS", "RT"): ("RUS (auto) on RT", (4, -17)), ("NM3", "DRY"): ("NearMiss-3 on DRY", (13, -44)),
        ("NearMiss", "CM"): ("NearMiss-1 on CM", (13, -39)), ("ENN", "CAR"): ("ENN on CAR", (50, -36))}
for (mt, ds), (txt, xy) in NOTE.items():
    r = m[(m.method == mt) & (m.dataset == ds)].iloc[0]
    ax.annotate(txt, (r.kept, r.delta), xytext=xy, textcoords="data", fontsize=6.6, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=3))
ax.set_xlim(-3, 103); ax.set_ylim(-48, 4); ax.set_xlabel("Local training data kept on the clients [%]")
ax.set_ylabel("Difference in test F1-macro\nto the baseline [%]")
ax.legend(frameon=False, fontsize=6.6, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
fig.savefig(OUT / "figH_kept_vs_delta.pdf"); fig.savefig(OUT / "figH_kept_vs_delta.png"); print(round(rho, 3))
