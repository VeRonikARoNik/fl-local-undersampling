#!/usr/bin/env python3
"""The moment of undersampling: the three paths of the original experiment.

  baseline   no undersampling
  us_after   undersampling on each client after the client partition (the design analysed in the paper)
  us_before  undersampling of the whole data set before the cross-validation split (path US->FL); its validation and
             test halves are taken from the undersampled data, so it is evaluated on different test data

Inputs: fl_results_clean.csv (baseline, us_after with the 360 corrected runs), the stored results of all paths
(../results/fl_results_raw.csv, or the file named by the environment variable RAW) and the class counts of the original split files counted on the machine that
ran the experiment (path1_audit_<CODE>.csv from audit_path1.py: test files and the training part at K = 3).
Outputs: paths_by_method.csv, paths_by_dataset_method.csv, path1_testsets.csv"""
import glob, os, sys
import numpy as np, pandas as pd


def first_existing(*paths):
    """The package layout first (analysis/ next to results/), then the layout of the working folder."""
    return next((p for p in paths if os.path.exists(p)), paths[0])


AUD = sys.argv[1] if len(sys.argv) > 1 else first_existing("../results/path1_audit", "path1_audit")
METHODS = ["RUS75", "RUS50", "RUS25", "TomekLinks", "AllKNN", "RUS", "ENN", "NearMiss", "NM3"]
clean = pd.read_csv("fl_results_clean.csv")
raw = pd.read_csv(os.environ.get("RAW", first_existing("../results/fl_results_raw.csv",
                                                       "../repro/results/fl_results_raw.csv")))
raw = raw[(raw.dataset != "SOY") & (raw.us_method != "NCL") & (raw.fl_algorithm != "FedBagging")]
before = raw[raw.path_type == "us_before"]
assert len(before) == 9 * 8640, len(before)

f1 = lambda d: d.test_f1_macro.mean() * 100
rows = []
b0 = f1(clean[clean.us_method == "none"])
for m in METHODS:
    rows.append((m, b0, f1(clean[clean.us_method == m]), f1(before[before.us_method == m]),
                 f1(raw[(raw.path_type == "us_after") & (raw.us_method == m)])))
t = pd.DataFrame(rows, columns=["method", "baseline", "after_split", "before_split", "after_split_stored"])
t["after_minus_base"] = t.after_split - t.baseline
t["before_minus_base"] = t.before_split - t.baseline
t.to_csv("paths_by_method.csv", index=False)
print(t.round(2).to_string(index=False))

rows = []
for ds in sorted(clean.dataset.unique()):
    bd = f1(clean[(clean.us_method == "none") & (clean.dataset == ds)])
    for m in METHODS:
        rows.append((ds, m, bd, f1(clean[(clean.us_method == m) & (clean.dataset == ds)]),
                     f1(before[(before.us_method == m) & (before.dataset == ds)])))
pd.DataFrame(rows, columns=["dataset", "method", "baseline", "after_split", "before_split"]).to_csv(
    "paths_by_dataset_method.csv", index=False)

# ---- test sets and training parts of path US->FL compared with the baseline split files ----
aud = pd.concat([pd.read_csv(f, dtype={"cls": str}) for f in sorted(glob.glob(f"{AUD}/path1_audit_*.csv"))])
assert (aud.cls != "__missing__").all()
rows_ = aud[aud.cls == "__rows__"].pivot_table(index=["dataset", "fold", "part"], columns="method", values="n")
ncls = aud[aud.cls != "__rows__"].query("n > 0").groupby(["dataset", "method", "fold", "part"]).cls.nunique()
ncls = ncls.unstack("method")
minc = aud[aud.cls != "__rows__"].query("n > 0").groupby(["dataset", "method", "fold", "part"]).n.min().unstack("method")
out = []
for (ds, fold, part), r in rows_.iterrows():
    for m in METHODS:
        out.append((ds, fold, part, m, r["none"], r[m], 100 * r[m] / r["none"],
                    ncls.loc[(ds, fold, part), "none"], ncls.loc[(ds, fold, part), m],
                    minc.loc[(ds, fold, part), "none"], minc.loc[(ds, fold, part), m]))
o = pd.DataFrame(out, columns=["dataset", "fold", "part", "method", "n_baseline", "n_before", "kept_pct",
                               "classes_baseline", "classes_before", "smallest_baseline", "smallest_before"])
o.to_csv("path1_testsets.csv", index=False)
s = (o[o.part == "test"].groupby("method")
     .agg(kept_pct_mean=("kept_pct", "mean"), kept_pct_min=("kept_pct", "min"), kept_pct_max=("kept_pct", "max"))
     .reindex(METHODS))
print("\nshare of the baseline test half kept in path US->FL (mean over data sets and folds):")
print(s.round(1).to_string())
print("\ntest halves with fewer classes than the baseline:",
      int(((o.part == "test") & (o.classes_before < o.classes_baseline)).sum()), "of", int((o.part == "test").sum()))
ds_kept = o[o.part == "test"].groupby(["method", "dataset"]).kept_pct.mean().unstack().reindex(METHODS)
print(ds_kept.round(1).to_string())
