"""Shared style for the figures of the manuscript (print-oriented, recessive chrome, honest axes)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, NEUTRAL, GREY = "#e34948", "#f0efec", "#b9b7b1"
LIGHTBLUE = "#a9c9f0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.unicode_minus": True,
})
US_LABEL = {"RUS": "RUS (auto)", "RUS25": "RUS 25%", "RUS50": "RUS 50%", "RUS75": "RUS 75%",
            "NearMiss": "NearMiss-1", "NM3": "NearMiss-3", "ENN": "ENN", "AllKNN": "AllKNN",
            "TomekLinks": "Tomek Links", "none": "Baseline"}
FL_LABEL = {"FedAvg": "FedAvg", "WeightedFedAvg": "Weighted FedAvg", "FedMedian": "Coordinate-wise median",
            "FedTrimmedMean": "Trimmed mean", "FedProx": "Distance-weighted", "FedBoost": "Rank-weighted"}
LEARNER_LABEL = {"DT": "Decision tree", "RF": "Random forest", "GB": "Gradient boosting", "XGB": "XGBoost"}
DATASETS = ["UNSW", "RT", "LR", "DRY", "CM", "SLS", "CAR", "SVS", "AIH"]
METHODS = ["RUS75", "RUS50", "RUS25", "TomekLinks", "AllKNN", "RUS", "ENN", "NearMiss", "NM3"]
LEARNERS = ["DT", "RF", "GB", "XGB"]
MINUS = "−"


def num(v, d=1, sign=False):
    """Format a number with a typographic minus sign and no negative zero."""
    r = round(float(v), d)
    if r == 0:
        r = 0.0
    s = f"{r:+.{d}f}" if sign else f"{r:.{d}f}"
    if sign and r == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", MINUS)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)
    print("saved", name)
