#!/usr/bin/env python3
"""How the elements of the study affect test F1-macro and each other (Section "How the elements interact").

Inputs: fl_results_clean.csv (86 400 runs: baseline and per-client undersampling), data_summary.csv and
client_smallest_median.csv (data set table and client audit).
Outputs:
  main_effects.csv           mean F1 (%) per level of every factor, over all other factors
  anova_all2way.csv          type II ANOVA with the five main effects and all ten two-way interactions (omega squared)
  ix_method_rule.csv         mean difference to the baseline (%) per undersampling method and aggregation rule
  ix_method_K.csv            ... per method and number of clients K
  ix_method_learner.csv      ... per method and learner
  ix_rule_dataset.csv        mean difference to FedAvg (%) per aggregation rule and data set
  ix_rule_K.csv              ... per rule and K
  ix_rule_method.csv         ... per rule and undersampling condition (none = baseline)
  ix_dataset_properties.csv  per data set: mean change by the five harmful methods, smallest class per client at K = 10
                             (median), baseline F1, imbalance ratio and size; Spearman correlations are printed
Differences are taken over matched runs (same data set, fold, K, learner and rule, or method for the rules)."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

Y = "test_f1_macro"
df = pd.read_csv("fl_results_clean.csv")
df["F1"] = df[Y] * 100

# ---- main effects ----
rows = []
for fac in ["dataset", "us_method", "local_model", "fl_algorithm", "n_clients"]:
    for lev, v in df.groupby(fac)["F1"].mean().items():
        rows.append((fac, str(lev), v))
me = pd.DataFrame(rows, columns=["factor", "level", "mean_f1"])
me.to_csv("main_effects.csv", index=False)
print(me.groupby("factor").mean_f1.agg(["min", "max"]).assign(range=lambda d: d["max"] - d["min"]).round(2))

# ---- ANOVA with all ten two-way interactions ----
d2 = df.assign(US=df.us_method, K=df.n_clients.astype(str), L=df.local_model, A=df.fl_algorithm, D=df.dataset)
f = (f"{Y} ~ C(D) + C(US) + C(L) + C(A) + C(K) + C(D):C(US) + C(D):C(L) + C(D):C(A) + C(D):C(K) + C(US):C(L)"
     " + C(US):C(A) + C(US):C(K) + C(L):C(A) + C(L):C(K) + C(A):C(K)")
m = smf.ols(f, data=d2).fit()
a = anova_lm(m, typ=2)
ms_res = a.loc["Residual", "sum_sq"] / a.loc["Residual", "df"]
sst = ((d2[Y] - d2[Y].mean()) ** 2).sum()
a["omega2"] = (a.sum_sq - a.df * ms_res) / (sst + ms_res)
a["partial_eta2"] = a.sum_sq / (a.sum_sq + a.loc["Residual", "sum_sq"])
a.loc["Residual", ["omega2", "partial_eta2"]] = np.nan
a = a.sort_values("sum_sq", ascending=False)
a.to_csv("anova_all2way.csv")
pd.set_option("display.width", 200)
print(a.round(4).to_string()); print("R2", round(m.rsquared, 4))

# ---- interactions as matched differences ----
key = ["dataset", "fold", "n_clients", "local_model", "fl_algorithm"]
base = df[df.us_method == "none"].set_index(key)["F1"].rename("F1_base")
us = df[df.us_method != "none"].join(base, on=key)
us["delta"] = us.F1 - us.F1_base
for other, name in [("fl_algorithm", "rule"), ("n_clients", "K"), ("local_model", "learner")]:
    t = us.groupby(["us_method", other]).delta.mean().unstack()
    t.to_csv(f"ix_method_{name}.csv")
    print(f"\nmethod x {name}: spread of the method effect across levels (max - min)")
    print((t.max(axis=1) - t.min(axis=1)).round(2).to_dict())

rkey = ["dataset", "fold", "n_clients", "local_model", "us_method"]
fa = df[df.fl_algorithm == "FedAvg"].set_index(rkey)["F1"].rename("F1_fedavg")
ru = df[df.fl_algorithm != "FedAvg"].join(fa, on=rkey)
ru["delta"] = ru.F1 - ru.F1_fedavg
for other, name in [("dataset", "dataset"), ("n_clients", "K"), ("us_method", "method")]:
    t = ru.groupby(["fl_algorithm", other]).delta.mean().unstack()
    t.to_csv(f"ix_rule_{name}.csv")
    print(f"\nrule x {name}: min / max of the mean difference to FedAvg")
    print(t.min(axis=1).round(2).to_dict(), t.max(axis=1).round(2).to_dict())

# ---- which properties of a data set go with the losses of the harmful methods (nine points: descriptive) ----
HARMFUL = ["AllKNN", "RUS", "ENN", "NearMiss", "NM3"]
summ = pd.read_csv("data_summary.csv").set_index("code")
small = pd.read_csv("client_smallest_median.csv").set_index("dataset")
props = pd.DataFrame({
    "harm_mean": us[us.us_method.isin(HARMFUL)].groupby("dataset").delta.mean(),
    "smallest_K10": small["10"],
    "baseline": df[df.us_method == "none"].groupby("dataset")["F1"].mean(),
    "IR": summ["IR"], "n": summ["instances"]}).dropna()
props.to_csv("ix_dataset_properties.csv")
for c in ["IR", "smallest_K10", "n", "baseline"]:
    r = stats.spearmanr(props[c], props.harm_mean)
    print(f"loss of the harmful methods vs {c}: Spearman rho = {r.correlation:.2f}, p = {r.pvalue:.3f}")
r = stats.spearmanr(props.IR, props.smallest_K10)
print(f"IR vs smallest class at K = 10: Spearman rho = {r.correlation:.2f}, p = {r.pvalue:.3f}")

