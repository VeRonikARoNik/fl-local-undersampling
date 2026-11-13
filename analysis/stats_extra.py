#!/usr/bin/env python3
"""Additional statistics for the Statistical analysis chapter.

1. ANOVA residual diagnostics (skewness, kurtosis, Jarque-Bera, Brown-Forsythe/Levene by method and by data set).
2. Robustness check: the same factorial model fitted to rank-transformed F1 (Conover and Iman), omega squared per term.
3. Friedman tests with the Iman-Davenport correction for every factor, including the number of clients K (baseline).
4. Wilcoxon signed-rank statistics (W, number of non-zero pairs) for the methods versus the baseline.
5. Aggregation rules versus FedAvg: paired differences, cluster-bootstrap CIs, Wilcoxon with Holm correction.
Inputs: fl_results_clean.csv, wilcoxon_vs_baseline_holm.csv. Outputs: see the files written at the end.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from scipy import stats

Y = "test_f1_macro"
df = pd.read_csv("fl_results_clean.csv")
df["US"] = df.us_method; df["K"] = df.n_clients.astype(str)
RHS = ("C(dataset) + C(US) + C(local_model) + C(fl_algorithm) + C(K)"
       " + C(US):C(local_model) + C(US):C(dataset) + C(local_model):C(dataset)"
       " + C(US):C(K) + C(US):C(fl_algorithm) + C(dataset):C(K)")


def anova(yname):
    m = smf.ols(f"{yname} ~ {RHS}", data=df).fit()
    a = anova_lm(m, typ=2)
    ss_res = a.loc["Residual", "sum_sq"]; ms_res = ss_res / a.loc["Residual", "df"]
    sst = ((df[yname] - df[yname].mean()) ** 2).sum()
    a["partial_eta2"] = a.sum_sq / (a.sum_sq + ss_res)
    a["omega2"] = (a.sum_sq - a.df * ms_res) / (sst + ms_res)
    a.loc["Residual", ["partial_eta2", "omega2"]] = np.nan
    return m, a


out = {}
# ---------- 1. residual diagnostics ----------
m, a = anova(Y)
r = m.resid.values; f = m.fittedvalues.values
out["R2"] = m.rsquared
out["resid_skew"] = float(stats.skew(r)); out["resid_excess_kurtosis"] = float(stats.kurtosis(r))
jb = stats.jarque_bera(r); out["jarque_bera"] = [float(jb.statistic), float(jb.pvalue)]
bf_us = stats.levene(*[g[Y].values for _, g in df.groupby("US")], center="median")
bf_res_us = stats.levene(*[r[(df.US == u).values] for u in df.US.unique()], center="median")
bf_res_ds = stats.levene(*[r[(df.dataset == d).values] for d in df.dataset.unique()], center="median")
out["brown_forsythe_F1_by_method"] = [float(bf_us.statistic), float(bf_us.pvalue)]
out["brown_forsythe_resid_by_method"] = [float(bf_res_us.statistic), float(bf_res_us.pvalue)]
out["brown_forsythe_resid_by_dataset"] = [float(bf_res_ds.statistic), float(bf_res_ds.pvalue)]
sd = pd.DataFrame({"resid": r * 100, "US": df.US.values, "dataset": df.dataset.values})
sd.groupby("US").resid.std().rename("resid_sd_pct").to_csv("resid_sd_by_method.csv")
sd.groupby("dataset").resid.std().rename("resid_sd_pct").to_csv("resid_sd_by_dataset.csv")
np.savez_compressed("anova_residuals.npz", resid=r.astype(np.float32), fitted=f.astype(np.float32))
out["resid_sd_overall_pct"] = float(r.std(ddof=1) * 100)
out["share_abs_resid_gt_10pct"] = float((np.abs(r) > 0.10).mean())

# ---------- 2. rank-transformed ANOVA ----------
df["rank_f1"] = stats.rankdata(df[Y]) / len(df)
_, ar = anova("rank_f1")
comp = pd.DataFrame({"omega2_raw": a.omega2, "omega2_rank": ar.omega2, "F_rank": ar.F, "p_rank": ar["PR(>F)"]}).drop("Residual")
comp = comp.sort_values("omega2_raw", ascending=False)
comp.to_csv("anova_raw_vs_rank.csv")
out["spearman_omega2_raw_vs_rank"] = float(stats.spearmanr(comp.omega2_raw, comp.omega2_rank).statistic)
out["rank_model_R2"] = float(smf.ols(f"rank_f1 ~ {RHS}", data=df).fit().rsquared)

# ---------- 3. Friedman + Iman-Davenport ----------
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
fr = []
for name, factor, data in [("undersampling method", "us_method", df), ("learner (baseline)", "local_model", df[df.us_method == "none"]),
                           ("aggregation rule", "fl_algorithm", df), ("number of clients K (baseline)", "n_clients", df[df.us_method == "none"])]:
    t = data.groupby(["dataset", factor])[Y].mean().unstack()
    N, k = t.shape
    chi2, p = stats.friedmanchisquare(*[t[c] for c in t.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    ranks = t.rank(axis=1, ascending=False).mean().sort_values()
    fr.append(dict(factor=name, k=k, N=N, chi2=chi2, df=k - 1, p=p, F_ID=ff, df1=k - 1, df2=(k - 1) * (N - 1), p_ID=p_ff,
                   q05=Q05[k], CD=cd, ranks=json.dumps({str(i): round(v, 3) for i, v in ranks.items()})))
    if factor == "n_clients":
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(t.values); nem.index = nem.columns = t.columns
        nem.round(4).to_csv("nemenyi_K_baseline.csv"); ranks.to_csv("meanranks_K_baseline.csv")
pd.DataFrame(fr).to_csv("friedman_summary.csv", index=False)

# ---------- 4. Wilcoxon statistics, methods vs baseline ----------
keys = ["dataset", "fold", "n_clients", "fl_algorithm", "local_model"]
piv = df.pivot_table(index=keys, columns="us_method", values=Y, aggfunc="first")
w = pd.read_csv("wilcoxon_vs_baseline_holm.csv").set_index("us_method")
rows = []
for meth in w.index:
    d = piv[meth] - piv["none"]
    res = stats.wilcoxon(piv[meth], piv["none"], zero_method="wilcox")
    rows.append(dict(us_method=meth, W=res.statistic, n_nonzero=int((d != 0).sum()), n_zero=int((d == 0).sum()),
                     share_better=float((d > 0).mean())))
wx = w.join(pd.DataFrame(rows).set_index("us_method"))
wx.to_csv("wilcoxon_vs_baseline_holm_full.csv")

# ---------- 5. rules vs FedAvg ----------
keys_r = ["dataset", "fold", "n_clients", "us_method", "local_model"]
pr = df.pivot_table(index=keys_r, columns="fl_algorithm", values=Y, aggfunc="first")
rng = np.random.default_rng(42)
rows = []
for rule in ["WeightedFedAvg", "FedProx", "FedTrimmedMean", "FedBoost", "FedMedian"]:
    d = (pr[rule] - pr["FedAvg"]) * 100
    per_ds = d.groupby(level="dataset").mean()
    bs = rng.choice(per_ds.values, size=(10000, len(per_ds))).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    nz = d[d != 0]
    res = stats.wilcoxon(pr[rule], pr["FedAvg"], zero_method="wilcox")
    rk = stats.rankdata(np.abs(nz)); rbc = (rk[nz.values > 0].sum() - rk[nz.values < 0].sum()) / rk.sum()
    rows.append(dict(rule=rule, n_pairs=len(d), mean_delta=per_ds.mean(), ci_lo=lo, ci_hi=hi, share_worse=float((d < 0).mean()),
                     share_equal=float((d == 0).mean()), W=res.statistic, p=res.pvalue, rank_biserial=rbc))
rv = pd.DataFrame(rows); rv["p_holm"] = multipletests(rv.p, method="holm")[1]
rv.to_csv("rules_vs_fedavg.csv", index=False)

json.dump(out, open("stats_extra.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(comp.round(4).to_string())
print(pd.DataFrame(fr).drop(columns="ranks").round(4).to_string())
print(pd.DataFrame(fr)[["factor", "ranks"]].to_string())
print(wx.round(4).to_string())
print(rv.round(4).to_string())
