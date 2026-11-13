"""Class counts of the evaluation (test) files and of the training part (sum of the K = 3 client files) in the
original split folders: baseline (no undersampling) and path US->FL (undersampling of the whole data set before the
cross-validation split, folder undersampling_before_split). Output: path1_audit_<CODE>.csv
Usage (on the machine that holds the split folders): SPLITS=/path/to/data_splits python audit_path1.py AIH"""
import os, sys, time
import pandas as pd
BASE = os.path.expanduser(os.environ.get("SPLITS", "data_splits"))   # folder written by the split scripts
OUT = os.path.expanduser(os.environ.get("OUT", "."))
DS = {"AIH": ("Avian_Influenza_H5N1(AIH)", "outcome"), "CAR": ("Car_Evaluation(CAR)", "class"),
      "SVS": ("Statlog_Vehicle_Silhouettes(SVS)", "18"), "SLS": ("Statlog_Landsat_Satellite(SLS)", "36"),
      "CM": ("Crowdsourced_Mapping(CM)", "class"), "DRY": ("Dry_Bean(DRY)", "Class"),
      "LR": ("Letter_Recognition(LR)", "lettr"), "RT": ("RT-IoT2022(RT)", "Attack_type"), "UNSW": ("UNSW_NB15(UNSW)", "label")}
ALL = ["none", "RUS", "RUS25", "RUS50", "RUS75", "NearMiss", "NM3", "ENN", "AllKNN", "TomekLinks"]
code = sys.argv[1]
folder, lab = DS[code]
rows, t0 = [], time.time()
K = 3
for m in ALL:
    for i in range(1, 6):
        if m == "none":
            d = f"{BASE}/{folder}/f{code}_fold_{i}/{K}_federation/f{code}_fold_{i}_{K}_federation"
            files = {"test": d + "_test.csv", "train": [d + f"_tr_{t}.csv" for t in range(1, K + 1)]}
        else:
            d = f"{BASE}/undersampling_before_split/{m}/{folder}/f{code}_fold_{i}/{K}_federation/f{code}_fold_{i}_{K}_federation"
            files = {"test": d + f"_test_{m}.csv", "train": [d + f"_tr_{t}_{m}.csv" for t in range(1, K + 1)]}
        for part in ("test", "train"):
            fl = files[part] if isinstance(files[part], list) else [files[part]]
            if not all(os.path.exists(f) for f in fl):
                rows.append((code, m, i, part, "__missing__", 0)); continue
            vc = pd.concat([pd.read_csv(f, usecols=[lab], dtype=str)[lab] for f in fl]).value_counts()
            rows.append((code, m, i, part, "__rows__", int(vc.sum())))
            for c, n in vc.items():
                rows.append((code, m, i, part, c, int(n)))
pd.DataFrame(rows, columns=["dataset", "method", "fold", "part", "cls", "n"]).to_csv(f"{OUT}/path1_audit_{code}.csv", index=False)
print(code, len(rows), "rows", round(time.time() - t0, 1), "s")
