#!/usr/bin/env python3
"""Per-class effect of ENN on one client (fold 1, K = 3, client 1) of RT and SVS (Section 8 of the paper).
Uses the class counts of the original client files (orig_audit_all.csv); writes enn_examples.csv."""
import pandas as pd
o = pd.read_csv("orig_audit_all.csv", dtype={"cls": str})
o = o[(o.fold == 1) & (o.K == 3) & (o.client == 1) & o.dataset.isin(["RT", "SVS"]) & ~o.cls.str.startswith("__")]
b = o[o.method == "none"].rename(columns={"n": "before"})[["dataset", "cls", "before"]]
a = o[o.method == "ENN"].rename(columns={"n": "after"})[["dataset", "cls", "after"]]
t = b.merge(a, on=["dataset", "cls"], how="left").fillna({"after": 0})
t["after"] = t.after.astype(int); t["kept_pct"] = (100 * t.after / t.before).round(1)
t = t.sort_values(["dataset", "before"], ascending=[True, False])
t.to_csv("enn_examples.csv", index=False); print(t.to_string(index=False))
