#!/usr/bin/env python3
"""Per-class audit of per-client undersampling.

Re-creates every client set with the original split code (federated_data_split_path2.py) and seeds and records,
for every client and method, the number of instances of each class before and after undersampling.
Used for the figure on which classes each method removes.

Environment variables: SPLIT_SCRIPT, DATA_DIR, METHODS, FOLDS, TAG; data set codes as arguments.
Output: us_audit_classes_<codes>_<TAG>.csv (one row per client set, method and class)
"""
import os, sys, time, logging, warnings, importlib.util
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_order  # noqa: F401  (file order as on Windows, where the experiment ran)

warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
SPLIT = os.environ.get("SPLIT_SCRIPT", "../src/original/federated_data_split_path2.py")
DATA = os.environ.get("DATA_DIR", "../datasets_original/").rstrip("/") + "/"
spec = importlib.util.spec_from_file_location("p2", SPLIT)
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
FOLDER = {"RT": "rt-iot2022", "LR": "letter+recognition", "UNSW": "unsw-nb15", "SLS": "statlog+landsat+satellite",
          "CAR": "car+evaluation", "SVS": "statlog+vehicle+silhouettes", "DRY": "dry+bean+dataset",
          "CM": "crowdsourced+mapping", "AIH": "aviaere_influenza_fallliste"}
METHODS = os.environ.get("METHODS", "RUS,RUS25,RUS50,RUS75,NearMiss,NM3,ENN,AllKNN,TomekLinks").split(",")
FOLDS = [int(f) for f in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]
codes = sys.argv[1:] or list(FOLDER)
out = f"us_audit_classes_{'_'.join(codes)}_{os.environ.get('TAG', 'all')}.csv"
rows = []
for code in codes:
    t0 = time.time()
    df, name = p2.load_dataset(DATA + FOLDER[code]); lab = p2.detect_label_column(df, dataset_name=name)
    folds = p2.stratified_kfold_split(df, lab, 5, 42)
    for i in FOLDS:
        Ditr = pd.concat([folds[j] for j in range(5) if j != i - 1], ignore_index=True)
        for K in range(3, 11):
            subsets = p2.stratified_split(Ditr, lab, K, 42 + i * 10 + K)
            for t in range(1, K + 1):
                c = subsets[t - 1]
                before = c[lab].value_counts()
                for m in METHODS:
                    res = p2.apply_undersampling(c, lab, m, random_state=42 + i * 100 + K * 10 + t)
                    after = res[lab].value_counts()
                    fb = res is c
                    for cls, nb in before.items():
                        rows.append(dict(dataset=code, fold=i, K=K, client=t, method=m, cls=cls, n_before=int(nb),
                                         n_after=int(after.get(cls, 0)), fallback=fb))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(code, "done", round(time.time() - t0), "s", flush=True)
