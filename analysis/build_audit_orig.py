#!/usr/bin/env python3
"""Client audit from the client training files written by the original run.

Input
  orig_audit_all.csv   class counts of every original client file (compare_audit_orig.py; counted with audit_orig.py on
                       the machine that ran the experiment)
  us_audit_classes_*.csv, us_audit_RT_UNSW_nm.csv   re-created client sets; used only for the fallback flag (whether the
                       sampler could be applied), which the files themselves do not record
Output
  audit_classes_orig.csv  one row per client set, method and class (n_before, n_after, fallback)
  audit_sizes_orig.csv    one row per client set and method (n_before, n_after, fallback, n_classes, min_class)
"""
import glob
import pandas as pd

o = pd.read_csv("orig_audit_all.csv", dtype={"cls": str})
o = o[~o.cls.isin(["__rows__", "__missing__"])]
base = o[o.method == "none"].rename(columns={"n": "n_before"}).drop(columns="method")
meth = o[o.method != "none"].rename(columns={"n": "n_after"})
key = ["dataset", "fold", "K", "client"]
methods = sorted(meth.method.unique())
# every class present on the client before undersampling, for every method (n_after = 0 if the class was removed)
grid = base.merge(pd.DataFrame({"method": methods}), how="cross")
cls = grid.merge(meth, on=key + ["method", "cls"], how="left")
extra = meth.merge(base, on=key + ["cls"], how="left")
assert extra.n_before.notna().all(), "a class appears after undersampling that was not on the client before"
cls["n_after"] = cls.n_after.fillna(0).astype(int)

# fallback flag from the re-created client sets (only NearMiss-1 and NearMiss-3 ever fall back)
rec = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("us_audit_classes_*.csv"))], ignore_index=True)
fb = rec.drop_duplicates(key + ["method"])[key + ["method", "fallback"]]
nm = pd.read_csv("us_audit_RT_UNSW_nm.csv")[key + ["method", "fallback"]]
fb = pd.concat([fb, nm]).drop_duplicates(key + ["method"], keep="last")
cls = cls.merge(fb, on=key + ["method"], how="left")
assert not cls.loc[cls.method.isin(["NearMiss", "NM3"]), "fallback"].isna().any()
assert not rec.loc[~rec.method.isin(["NearMiss", "NM3"]), "fallback"].any()
cls["fallback"] = cls.fallback.fillna(False).astype(bool)
cls = cls[key + ["method", "cls", "n_before", "n_after", "fallback"]].sort_values(key + ["method", "cls"])
cls.to_csv("audit_classes_orig.csv", index=False)

# consistency: a client set on which the sampler fell back must be unchanged in the original files
chk = cls[cls.fallback]
print("fallback class rows:", len(chk), "unchanged:", bool((chk.n_before == chk.n_after).all()))

g = (cls.groupby(key + ["method"], sort=False)
     .agg(n_before=("n_before", "sum"), n_after=("n_after", "sum"), fallback=("fallback", "first"),
          n_classes=("cls", "size"), min_class=("n_before", "min")).reset_index())
g.to_csv("audit_sizes_orig.csv", index=False)
print("audit_sizes_orig.csv", len(g), "client sets x methods;", g.groupby("dataset").size().to_dict())
print((g.groupby(["method"]).apply(lambda d: 100 * d.n_after.sum() / d.n_before.sum(), include_groups=False)).round(1))
