import pandas as pd, numpy as np, statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
df = pd.read_csv("fl_results_clean.csv")
df["US"] = df.us_method; df["K"] = df.n_clients.astype(str)
y = "test_f1_macro"
formula = (f"{y} ~ C(dataset) + C(US) + C(local_model) + C(fl_algorithm) + C(K)"
           " + C(US):C(local_model) + C(US):C(dataset) + C(local_model):C(dataset)"
           " + C(US):C(K) + C(US):C(fl_algorithm) + C(dataset):C(K)")
m = smf.ols(formula, data=df).fit()
a = anova_lm(m, typ=2)
ss_res = a.loc["Residual","sum_sq"]; ms_res = a.loc["Residual","mean_sq"] if "mean_sq" in a else ss_res/a.loc["Residual","df"]
a["partial_eta2"] = a.sum_sq/(a.sum_sq+ss_res)
sst = ((df[y]-df[y].mean())**2).sum()
a["omega2"] = (a.sum_sq - a.df*ms_res)/(sst+ms_res)
a.loc["Residual",["partial_eta2","omega2"]] = np.nan
a = a.sort_values("sum_sq", ascending=False)
pd.set_option("display.width",200)
print(a.round(4).to_string()); print("R2", round(m.rsquared,4), "n", int(m.nobs))
a.to_csv("anova_typeII.csv")
r = m.resid
print("resid skew", round(stats.skew(r),3), "kurt", round(stats.kurtosis(r),3))
print("Levene across US (median):", stats.levene(*[g[y].values for _,g in df.groupby("US")], center="median"))
