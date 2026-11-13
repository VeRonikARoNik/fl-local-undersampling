#!/usr/bin/env python3
"""Compare the class counts of the client training files written by the original run (counted on the machine that ran
the experiment, orig_audit_*.csv) with the client sets re-created here (us_audit_classes_*.csv).

Writes
  orig_audit_all.csv            class counts of every original client file (dataset, method, fold, K, client, cls, n)
  audit_orig_vs_recreated.csv   per client set: rows in the original file, rows re-created, identical class counts
  audit_orig_vs_recreated_summary.csv   share of identical client sets per data set and method
"""
import sys
from pathlib import Path
import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")     # folder with the orig_audit_*.csv files
parts = [pd.read_csv(f, dtype={"cls": str}) for f in sorted(SRC.glob("orig_audit_*.csv")) if not f.stem.endswith("_t1")]
orig = pd.concat(parts, ignore_index=True).drop_duplicates()
orig.to_csv("orig_audit_all.csv", index=False)
print("original files:", orig[orig.cls == "__rows__"].shape[0], "client sets;",
      "missing:", int((orig.cls == "__missing__").sum()))

rec = pd.concat([pd.read_csv(f, dtype={"cls": str}) for f in sorted(Path(".").glob("us_audit_classes_*.csv"))],
                ignore_index=True)
key = ["dataset", "method", "fold", "K", "client"]
# baseline rows of the re-created audit are the n_before counts of any method (identical across methods)
base = (rec.drop_duplicates(["dataset", "fold", "K", "client", "cls"])
        .assign(method="none", n=lambda d: d.n_before)[key + ["cls", "n"]])
after = rec.assign(n=rec.n_after)[key + ["cls", "n"]]
recl = pd.concat([base, after], ignore_index=True)
recl = recl[recl.n > 0]

o = orig[~orig.cls.isin(["__rows__", "__missing__"])]
o = o[o.n > 0]


def signature(d):
    return (d.sort_values("cls").groupby(key)
            .apply(lambda g: ";".join(f"{c}:{n}" for c, n in zip(g.cls, g.n)), include_groups=False)
            .rename("sig"))


so = signature(o).rename("sig_orig")
sr = signature(recl).rename("sig_rec")
no = o.groupby(key).n.sum().rename("n_orig")
nr = recl.groupby(key).n.sum().rename("n_rec")
cmp = pd.concat([so, sr, no, nr], axis=1, join="inner").reset_index()
cmp["identical"] = cmp.sig_orig == cmp.sig_rec
cmp["abs_diff_rows"] = (cmp.n_orig - cmp.n_rec).abs()
cmp["rel_diff_pct"] = 100 * cmp.abs_diff_rows / cmp.n_orig
# class-wise difference: sum over classes of |n_original - n_recreated|, relative to the size of the original set
cw = (o.merge(recl, on=key + ["cls"], how="outer", suffixes=("_o", "_r")).fillna({"n_o": 0, "n_r": 0})
      .assign(ad=lambda d: (d.n_o - d.n_r).abs()).groupby(key).ad.sum().rename("abs_diff_classes"))
cmp = cmp.merge(cw.reset_index(), on=key, how="left")
cmp["classwise_diff_pct"] = 100 * cmp.abs_diff_classes / cmp.n_orig
cmp.drop(columns=["sig_orig", "sig_rec"]).to_csv("audit_orig_vs_recreated.csv", index=False)

summ = (cmp.groupby(["dataset", "method"])
        .agg(client_sets=("identical", "size"), identical_pct=("identical", lambda s: 100 * s.mean()),
             max_abs_diff_rows=("abs_diff_rows", "max"), max_rel_diff_pct=("rel_diff_pct", "max"),
             median_classwise_diff_pct=("classwise_diff_pct", "median"),
             rows_orig=("n_orig", "sum"), rows_rec=("n_rec", "sum"))
        .reset_index())
summ.to_csv("audit_orig_vs_recreated_summary.csv", index=False)
pd.set_option("display.width", 200)
print(summ.pivot(index="method", columns="dataset", values="identical_pct").round(1))
print("compared client sets:", len(cmp), "identical:", round(100 * cmp.identical.mean(), 2), "%")
dif = cmp[~cmp.identical]
print("differing client sets:", len(dif), "median class-wise difference:", round(dif.classwise_diff_pct.median(), 2), "%",
      "max:", round(dif.classwise_diff_pct.max(), 1), "%")
print("rows orig vs rec (all):", cmp.n_orig.sum(), cmp.n_rec.sum())
