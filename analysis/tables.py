import pandas as pd, numpy as np
df=pd.read_csv("fl_results_clean.csv"); y="test_f1_macro"
rng=np.random.default_rng(42)
keys=["dataset","fold","n_clients","fl_algorithm","local_model"]
p=df.pivot_table(index=keys,columns="us_method",values=y,aggfunc="first")
d=(p.drop(columns="none").sub(p["none"],axis=0)*100); dds=d.groupby(level="dataset").mean()
w=pd.read_csv("wilcoxon_vs_baseline_holm.csv").set_index("us_method")
t=df[df.fl_algorithm=="FedAvg"].pivot_table(index=["dataset","fold","n_clients","local_model"],columns="us_method",values="train_time_sec",aggfunc="first")
sav=(1-t.drop(columns="none").div(t["none"],axis=0)).groupby(level="dataset").median().mean()*100
ranks=pd.read_csv("meanranks_us_method.csv",index_col=0).iloc[:,0]
rows=[]
for m in dds.columns:
    bs=rng.choice(dds[m].values,size=(10000,9)).mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
    rows.append(dict(method=m, mean_F1=100*df[df.us_method==m][y].mean(), delta=dds[m].mean(), ci_lo=lo, ci_hi=hi,
        p_holm=w.loc[m,"p_holm"], rbc=w.loc[m,"rank_biserial"], share_worse=100*w.loc[m,"share_worse"],
        time_saving=sav[m], mean_rank=ranks[m]))
T=pd.DataFrame(rows).sort_values("delta",ascending=False)
base=dict(method="none",mean_F1=100*df[df.us_method=="none"][y].mean(),mean_rank=ranks["none"])
T=pd.concat([pd.DataFrame([base]),T]); T.round(4).to_csv("table_us_methods.csv",index=False); print(T.round(2).to_string(index=False))
