"""Exact duplicates across the split (Section 12.1, threats to validity): for every data set and fold, the share of the
rows of the test file that also occur in the training part of the same fold (the sum of the K = 3 client files of the
baseline split, before undersampling), with the same feature values and label, and with the same feature values only.
The test file and the training part are the same for every number of clients.
Output: duplicates_split.csv (one row per data set and fold)
Usage (on the machine that holds the split folders): SPLITS=/path/to/data_splits python duplicates_split.py [CODE ...]"""
import os, sys, time
import pandas as pd
BASE = os.path.expanduser(os.environ.get("SPLITS", "data_splits"))   # folder written by the split scripts
OUT = os.path.expanduser(os.environ.get("OUT", "."))
DS = {"UNSW": ("UNSW_NB15(UNSW)", "label"), "RT": ("RT-IoT2022(RT)", "Attack_type"), "LR": ("Letter_Recognition(LR)", "lettr"),
      "DRY": ("Dry_Bean(DRY)", "Class"), "CM": ("Crowdsourced_Mapping(CM)", "class"),
      "SLS": ("Statlog_Landsat_Satellite(SLS)", "36"), "CAR": ("Car_Evaluation(CAR)", "class"),
      "SVS": ("Statlog_Vehicle_Silhouettes(SVS)", "18"), "AIH": ("Avian_Influenza_H5N1(AIH)", "outcome")}
K = 3
rows = []
for code in sys.argv[1:] or list(DS):
    folder, lab = DS[code]; t0 = time.time()
    for i in range(1, 6):
        d = f"{BASE}/{folder}/f{code}_fold_{i}/{K}_federation/f{code}_fold_{i}_{K}_federation"
        # read as text: rows written from the same table are equal exactly when their text is equal
        test = pd.read_csv(d + "_test.csv", dtype=str, keep_default_na=False)
        train = pd.concat([pd.read_csv(d + f"_tr_{t}.csv", dtype=str, keep_default_na=False) for t in range(1, K + 1)],
                          ignore_index=True)
        assert list(test.columns) == list(train.columns) and lab in test.columns
        feats = [c for c in test.columns if c != lab]
        h = lambda df, cols: pd.util.hash_pandas_object(df[cols], index=False)
        row_in = h(test, feats + [lab]).isin(h(train, feats + [lab]).values)
        feat_in = h(test, feats).isin(h(train, feats).values)
        rows.append(dict(dataset=code, fold=i, test_rows=len(test), train_rows=len(train),
                         test_rows_in_train_pct=round(100 * row_in.mean(), 2),
                         test_features_in_train_pct=round(100 * feat_in.mean(), 2)))
    print(code, round(time.time() - t0, 1), "s", flush=True)
r = pd.DataFrame(rows)
r.to_csv(f"{OUT}/duplicates_split.csv", index=False)
print(r.groupby("dataset", sort=False)[["test_rows_in_train_pct", "test_features_in_train_pct"]].mean().round(1).to_string())
