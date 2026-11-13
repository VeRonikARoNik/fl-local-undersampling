"""Table 3: share of local data kept and share of clients on which the method could not be applied.
Input: audit_sizes_orig.csv (build_audit_orig.py): every client training file of the original run, all data sets,
folds, K and methods."""
import pandas as pd

a = pd.read_csv("audit_sizes_orig.csv")
a["kept"] = a.n_after / a.n_before
per_ds = a.groupby(["method", "dataset"]).agg(kept=("kept", "mean"), fallback=("fallback", "mean"), n=("fallback", "size"))
per_ds.to_csv("us_audit_per_dataset.csv")
rows = []
for m, g in per_ds.groupby(level="method"):
    g = g.droplevel("method")
    fb = g[g.fallback > 0].fallback.sort_values(ascending=False)
    rows.append(dict(method=m, kept_mean=100 * g.kept.mean(), kept_min=100 * g.kept.min(), kept_max=100 * g.kept.max(),
                     fallback_overall=100 * (a[a.method == m].fallback.mean()),
                     fallback_datasets="; ".join(f"{d} {100*v:.0f}%" for d, v in fb.items()) or "none"))
t = pd.DataFrame(rows).sort_values("kept_mean", ascending=False)
t.round(1).to_csv("table3_us_audit.csv", index=False)
print(t.round(1).to_string(index=False))
print("\nclients per dataset/method:", per_ds.n.unique())
