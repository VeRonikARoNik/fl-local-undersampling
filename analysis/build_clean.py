"""Build the clean analysis dataset from fl_results.csv.

Exclusions (documented in Section "Problems encountered" of the paper and in the README):
  - path_type == us_before : US applied to whole dataset incl. val/test (non-comparable test sets)
  - us_method == NCL       : sampler never executed (SciPy 1.17 incompatibility), rows equal baseline
  - dataset == SOY         : baseline split loaded non-data description files as rows
  - fl_algorithm == FedBagging : numerically identical to FedAvg (100% of rows)

Correction:
  - In 15 federations (ENN: 14 on AIH, 1 on RT) a class was absent from every client after undersampling. fl_training.py
    coded the labels over the clients' training labels only, so the validation and test instances of that class were
    scored as the first class and the class dropped out of the macro averages. The metrics of these 360 runs are
    replaced by the values of rescore_vanished.py, which scores the same predictions over all classes
    (rescored_vanished_runs.csv). The stored values are kept in fl_results_raw.csv and in that file.
"""
import os
import pandas as pd

src = os.environ.get("RAW", "../results/fl_results_raw.csv")   # all runs as written by fl_training.py
df = pd.read_csv(src)
n0 = len(df)
clean = df[(df.path_type != "us_before") & (df.us_method != "NCL") &
           (df.dataset != "SOY") & (df.fl_algorithm != "FedBagging")].copy()

# --- correction of the runs in which a class vanished from every client ---
fix = pd.read_csv("rescored_vanished_runs.csv")
metrics = [c[:-5] for c in fix.columns if c.endswith("_corr")]           # e.g. test_f1_macro
fix = fix.rename(columns={"K": "n_clients"})
key = ["dataset", "fold", "n_clients", "us_method", "fl_algorithm", "local_model"]
idx = clean.reset_index().merge(fix[key + [m + "_corr" for m in metrics]], on=key, how="inner")
assert len(idx) == len(fix) and (clean.loc[idx["index"], "path_type"] == "us_after").all()
for m in metrics:
    clean.loc[idx["index"].values, m] = idx[m + "_corr"].values
print("corrected", len(idx), "runs:", fix.groupby(["dataset", "us_method"]).size().to_dict())

clean.to_csv("fl_results_clean.csv", index=False)
print(n0, "->", len(clean))
print(clean.groupby(["path_type", "us_method"]).size().unstack(0))
