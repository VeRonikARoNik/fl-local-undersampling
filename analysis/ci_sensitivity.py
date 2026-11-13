#!/usr/bin/env python3
"""Sensitivity of the data-set-level conclusions to the choice of interval (Section "Paired comparisons").

For every undersampling method (versus the baseline) and every aggregation rule (versus FedAvg) the paired differences
are averaged per data set (nine values). From these nine values the script computes
  the cluster-bootstrap percentile interval used in the paper (10 000 resamples, seed 42, as in tables.py),
  the bias-corrected and accelerated (BCa) bootstrap interval,
  the t interval with 8 degrees of freedom,
  a Bonferroni-adjusted percentile interval (99.44% for the nine methods, 99% for the five rules),
  the exact Wilcoxon signed-rank test on the nine data-set means (two-sided).
Output: ci_sensitivity.csv
"""
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv("fl_results_clean.csv"); y = "test_f1_macro"
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
RULE_ORDER = ["WeightedFedAvg", "FedProx", "FedTrimmedMean", "FedBoost", "FedMedian"]   # as in stats_extra.py
for block, D, m_adj in (("method", d_methods, 9), ("rule", d_rules[RULE_ORDER], 5)):
    rng = np.random.default_rng(42)                          # the same draws as tables.py and stats_extra.py
    for c in D.columns:
        x = D[c].values
        bs = rng.choice(x, size=(10000, 9)).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lo_b, hi_b = np.percentile(bs, [100 * 0.025 / m_adj, 100 * (1 - 0.025 / m_adj)])
        bl, bh = bca(x)
        se = x.std(ddof=1) / np.sqrt(len(x)); t = stats.t.ppf(0.975, len(x) - 1)
        w = stats.wilcoxon(x, zero_method="wilcox", method="exact") if np.any(x != 0) else None
        rows.append(dict(block=block, level=c, mean=x.mean(), pct_lo=lo, pct_hi=hi, bca_lo=bl, bca_hi=bh,
                         t_lo=x.mean() - t * se, t_hi=x.mean() + t * se, bonf_lo=lo_b, bonf_hi=hi_b,
                         n_datasets_below=int((x < 0).sum()), wilcoxon_ds_p=w.pvalue if w else np.nan))
r = pd.DataFrame(rows)
r.to_csv("ci_sensitivity.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))
