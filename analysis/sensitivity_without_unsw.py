#!/usr/bin/env python3
"""Sensitivity of the conclusions to UNSW-NB15 (Section 12.1, threats to validity).

In UNSW-NB15, 45% of the test rows have an exact copy in the training part of the same fold (duplicates_split.csv),
so its absolute scores may be optimistic. This script repeats the data-set-level analysis without UNSW-NB15:
  for every undersampling method (versus the baseline) and every aggregation rule (versus FedAvg), the mean of the
  eight data-set means of the paired differences, the cluster-bootstrap percentile interval (10 000 resamples,
  seed 42, as in ci_sensitivity.py), the BCa and t intervals, the Bonferroni-adjusted percentile interval (99.44% for
  the nine methods, 99% for the five rules), the number of data sets below zero and the exact Wilcoxon signed-rank
  test on the eight data-set means;
  the factorial ANOVA of anova.py (type II, omega squared) on the remaining 76 800 runs.
Output: sensitivity_without_unsw.csv, anova_typeII_without_unsw.csv
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats

LEFT_OUT = "UNSW"
df = pd.read_csv("fl_results_clean.csv")
df = df[df.dataset != LEFT_OUT].copy()
y = "test_f1_macro"
keys = ["dataset", "fold", "n_clients", "fl_algorithm", "local_model"]
p = df.pivot_table(index=keys, columns="us_method", values=y, aggfunc="first")
d_methods = (p.drop(columns="none").sub(p["none"], axis=0) * 100).groupby(level="dataset").mean()
keys_r = ["dataset", "fold", "n_clients", "us_method", "local_model"]
q = df.pivot_table(index=keys_r, columns="fl_algorithm", values=y, aggfunc="first")
d_rules = (q.drop(columns="FedAvg").sub(q["FedAvg"], axis=0) * 100).groupby(level="dataset").mean()


def bca(x, B=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x); theta = x.mean()
    boot = rng.choice(x, size=(B, n)).mean(1)
    z0 = stats.norm.ppf(np.clip((boot < theta).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([np.delete(x, i).mean() for i in range(n)]); jm = jack.mean()
    a = ((jm - jack) ** 3).sum() / (6 * ((jm - jack) ** 2).sum() ** 1.5) if ((jm - jack) ** 2).sum() > 0 else 0.0
    out = []
    for z in (stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)):
        q_ = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(np.percentile(boot, 100 * q_))
    return out


rows = []
RULE_ORDER = ["WeightedFedAvg", "FedProx", "FedTrimmedMean", "FedBoost", "FedMedian"]   # as in ci_sensitivity.py
for block, D, m_adj in (("method", d_methods, 9), ("rule", d_rules[RULE_ORDER], 5)):
    rng = np.random.default_rng(42)
    for c in D.columns:
        x = D[c].values; n = len(x)
        bs = rng.choice(x, size=(10000, n)).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lo_b, hi_b = np.percentile(bs, [100 * 0.025 / m_adj, 100 * (1 - 0.025 / m_adj)])
        bl, bh = bca(x)
        se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
        w = stats.wilcoxon(x, zero_method="wilcox", method="exact") if np.any(x != 0) else None
        rows.append(dict(block=block, level=c, n_datasets=n, mean=x.mean(), pct_lo=lo, pct_hi=hi, bca_lo=bl, bca_hi=bh,
                         t_lo=x.mean() - t * se, t_hi=x.mean() + t * se, bonf_lo=lo_b, bonf_hi=hi_b,
                         n_datasets_below=int((x < 0).sum()), wilcoxon_ds_p=w.pvalue if w else np.nan))
r = pd.DataFrame(rows)
r.to_csv("sensitivity_without_unsw.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))

# the factorial ANOVA of anova.py without UNSW-NB15
df["US"] = df.us_method; df["K"] = df.n_clients.astype(str)
formula = (f"{y} ~ C(dataset) + C(US) + C(local_model) + C(fl_algorithm) + C(K)"
           " + C(US):C(local_model) + C(US):C(dataset) + C(local_model):C(dataset)"
           " + C(US):C(K) + C(US):C(fl_algorithm) + C(dataset):C(K)")
m = smf.ols(formula, data=df).fit()
a = anova_lm(m, typ=2)
ss_res = a.loc["Residual", "sum_sq"]; ms_res = ss_res / a.loc["Residual", "df"]
a["partial_eta2"] = a.sum_sq / (a.sum_sq + ss_res)
sst = ((df[y] - df[y].mean()) ** 2).sum()
a["omega2"] = (a.sum_sq - a.df * ms_res) / (sst + ms_res)
a.loc["Residual", ["partial_eta2", "omega2"]] = np.nan
a = a.sort_values("sum_sq", ascending=False)
a.to_csv("anova_typeII_without_unsw.csv")
print(a[["df", "omega2"]].round(4).to_string()); print("runs", int(m.nobs))
