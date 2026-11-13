import pandas as pd, numpy as np, scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
df = pd.read_csv("fl_results_clean.csv"); y="test_f1_macro"
out = {}
# --- Friedman + Nemenyi, blocks = datasets (Demsar 2006) ---
def friedman(factor, data):
    t = data.groupby(["dataset", factor])[y].mean().unstack()
    st, p = stats.friedmanchisquare(*[t[c] for c in t.columns])
    ranks = t.rank(axis=1, ascending=False).mean().sort_values()
    k, N = t.shape[1], t.shape[0]
    q = {4:2.569,6:2.850,10:3.164}[k]
    cd = q*np.sqrt(k*(k+1)/(6*N))
    nem = sp.posthoc_nemenyi_friedman(t.values); nem.index = nem.columns = t.columns
    print(f"\n== {factor}: Friedman chi2={st:.2f}, p={p:.2e}, N={N}, k={k}, CD(0.05)={cd:.2f}")
    print("mean ranks:", ranks.round(2).to_dict())
    return t, ranks, cd, nem, st, p
res = {}
for f in ["us_method","local_model","fl_algorithm"]:
    res[f] = friedman(f, df)
    res[f][3].round(4).to_csv(f"nemenyi_{f}.csv"); res[f][1].to_csv(f"meanranks_{f}.csv")
# learners compared on the baseline only (no undersampling), as reported in Section 5.4
rb = friedman("local_model", df[df.us_method == "none"])
rb[3].round(4).to_csv("nemenyi_local_model_baseline.csv"); rb[1].to_csv("meanranks_local_model_baseline.csv")
# --- Wilcoxon signed-rank: each US method vs baseline, paired on (dataset, fold, K, FL, model) ---
keys = ["dataset","fold","n_clients","fl_algorithm","local_model"]
piv = df.pivot_table(index=keys, columns="us_method", values=y, aggfunc="first")
rows = []
for m in [c for c in piv.columns if c != "none"]:
    d = (piv[m]-piv["none"]).dropna()
    w = stats.wilcoxon(piv[m], piv["none"], zero_method="wilcox")
    nz = d[d!=0]; rb = ( (nz>0).sum() - (nz<0).sum() )/len(nz)  # matched-pairs rank-biserial (sign proxy)
    # proper rank-biserial
    r = stats.rankdata(np.abs(nz)); rbc = (r[nz>0].sum()-r[nz<0].sum())/r.sum()
    rows.append(dict(us_method=m, n_pairs=len(d), mean_delta_pp=100*d.mean(), median_delta_pp=100*d.median(),
                     share_worse=(d<0).mean(), p=w.pvalue, rank_biserial=rbc))
w = pd.DataFrame(rows); w["p_holm"] = multipletests(w.p, method="holm")[1]
w = w.sort_values("mean_delta_pp", ascending=False)
print("\n== Wilcoxon vs baseline (us_after, Holm):"); print(w.round(4).to_string(index=False))
w.to_csv("wilcoxon_vs_baseline_holm.csv", index=False)
# --- per-dataset delta table (pp) ---
base = df[df.us_method=="none"].groupby("dataset")[y].mean()
tab = df[df.us_method!="none"].groupby(["dataset","us_method"])[y].mean().unstack()
delta = (tab.sub(base, axis=0)*100).round(1); delta.insert(0,"baseline_F1",base.round(3))
print("\n== per-dataset delta vs baseline (pp):"); print(delta.to_string())
delta.to_csv("per_dataset_delta_pp.csv")
# --- cost: training time ratio vs baseline (same dataset,fold,K,model; time identical across FL) ---
t = df[df.fl_algorithm=="FedAvg"].pivot_table(index=["dataset","fold","n_clients","local_model"], columns="us_method", values="train_time_sec", aggfunc="first")
ratio = t.drop(columns="none").div(t["none"], axis=0)
cost = ratio.groupby(level="dataset").median().round(2)
print("\n== median train-time ratio vs baseline (per dataset):"); print(cost.to_string())
print("overall median ratio:", ratio.median().round(2).to_dict())
cost.to_csv("time_ratio_vs_baseline.csv")
