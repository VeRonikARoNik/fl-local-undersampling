"""Re-create the per-client partitions of path 2 (same code, same seeds) and record, for every
client and undersampler: rows before/after, and whether the sampler fell back to the unmodified data."""
import logging, importlib.util, sys, time, warnings, pandas as pd
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import win_order  # noqa: F401  (data files in the order of the original Windows run)
spec = importlib.util.spec_from_file_location(
    "p2", _os.environ.get("SPLIT_SCRIPT", "../src/original/federated_data_split_path2.py"))
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
ORIG = _os.environ.get("DATA_DIR", "../datasets_original/").rstrip("/") + "/"
FOLD = {"RT":"rt-iot2022","LR":"letter+recognition","UNSW":"unsw-nb15","SLS":"statlog+landsat+satellite",
        "CAR":"car+evaluation","SVS":"statlog+vehicle+silhouettes","DRY":"dry+bean+dataset",
        "CM":"crowdsourced+mapping","AIH":"aviaere_influenza_fallliste"}
import os
METHODS = os.environ.get("METHODS","RUS,RUS25,RUS50,RUS75,NearMiss,NM3,ENN,AllKNN,TomekLinks").split(",")
FOLDS = [int(f) for f in os.environ.get("FOLDS","1,2,3,4,5").split(",")]
order = sys.argv[1:] or list(FOLD)
rows = []
for ac in order:
    df, name = p2.load_dataset(ORIG + FOLD[ac]); lab = p2.detect_label_column(df, dataset_name=name)
    folds = p2.stratified_kfold_split(df, lab, 5, 42)
    for i in FOLDS:
        Ditr = pd.concat([folds[j] for j in range(5) if j != i-1], ignore_index=True)
        for K in range(3, 11):
            subsets = p2.stratified_split(Ditr, lab, K, 42 + i*10 + K)
            for t in range(1, K+1):
                c = subsets[t-1]
                for m in METHODS:
                    t0 = time.time()
                    out = p2.apply_undersampling(c, lab, m, random_state=42 + i*100 + K*10 + t)
                    fell_back = out is c
                    rows.append(dict(dataset=ac, fold=i, K=K, client=t, method=m, n_before=len(c),
                                     n_after=len(out), fallback=fell_back, n_classes=c[lab].nunique(),
                                     min_class=int(c[lab].value_counts().min()), sec=round(time.time()-t0,3)))
    pd.DataFrame(rows).to_csv(f"us_audit_{'_'.join(order)}_{os.environ.get('TAG','all')}.csv", index=False)
    print(ac, "done", flush=True)
