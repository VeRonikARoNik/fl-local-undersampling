#!/usr/bin/env python3
"""Class counts of the client training files written by the original experiment (baseline and per-client undersampling).

Run on the machine that holds the split folders written by federated_data_split.py and federated_data_split_path2.py.
Usage: python audit_orig.py CODE [METHODS] [FOLDS] [TAG]
  CODE     data set code (AIH, CAR, SVS, SLS, CM, DRY, LR, RT, UNSW)
  METHODS  comma-separated conditions (default: none and all nine methods)
  FOLDS    comma-separated folds (default 1,2,3,4,5)
  TAG      suffix of the output file (default all)
Environment: SPLITS (folder data_splits, default ~/data_splits), OUT (output folder, default .)
Output: orig_audit_<CODE>_<TAG>.csv with dataset, method, fold, K, client, cls, n (cls "__rows__" = rows of the file)
"""
import os, sys, time
import pandas as pd

BASE = os.path.expanduser(os.environ.get("SPLITS", "~/data_splits"))
OUT = os.path.expanduser(os.environ.get("OUT", "."))
DS = {"AIH": ("Avian_Influenza_H5N1(AIH)", "outcome"), "CAR": ("Car_Evaluation(CAR)", "class"),
      "SVS": ("Statlog_Vehicle_Silhouettes(SVS)", "18"), "SLS": ("Statlog_Landsat_Satellite(SLS)", "36"),
      "CM": ("Crowdsourced_Mapping(CM)", "class"), "DRY": ("Dry_Bean(DRY)", "Class"),
      "LR": ("Letter_Recognition(LR)", "lettr"), "RT": ("RT-IoT2022(RT)", "Attack_type"), "UNSW": ("UNSW_NB15(UNSW)", "label")}
ALL = ["none", "RUS", "RUS25", "RUS50", "RUS75", "NearMiss", "NM3", "ENN", "AllKNN", "TomekLinks"]
code = sys.argv[1]
methods = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else ALL
folds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4, 5]
tag = sys.argv[4] if len(sys.argv) > 4 else "all"
folder, lab = DS[code]
rows, t0 = [], time.time()
for m in methods:
    for i in folds:
        for K in range(3, 11):
            for t in range(1, K + 1):
                if m == "none":
                    f = f"{BASE}/{folder}/f{code}_fold_{i}/{K}_federation/f{code}_fold_{i}_{K}_federation_tr_{t}.csv"
                else:
                    f = (f"{BASE}/undersampling_after_split/{m}/{folder}/f{code}_fold_{i}/{K}_federation/"
                         f"f{code}_fold_{i}_{K}_federation_tr_{t}_{m}.csv")
                if not os.path.exists(f):
                    rows.append((code, m, i, K, t, "__missing__", 0)); continue
                vc = pd.read_csv(f, usecols=[lab], dtype=str)[lab].value_counts()
                rows.append((code, m, i, K, t, "__rows__", int(vc.sum())))
                for c, n in vc.items():
                    rows.append((code, m, i, K, t, c, int(n)))
pd.DataFrame(rows, columns=["dataset", "method", "fold", "K", "client", "cls", "n"]).to_csv(
    f"{OUT}/orig_audit_{code}_{tag}.csv", index=False)
print(code, tag, len(rows), "rows", round(time.time() - t0, 1), "s")
