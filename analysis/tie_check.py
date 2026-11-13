#!/usr/bin/env python3
"""How often the neighbourhood used by the neighbour-based undersamplers is ambiguous (exact distance ties).

For the client sets of fold 1 with K = 5 (the configuration of the reproduction check), the features are coded as in
federated_data_split_path2.apply_undersampling (categorical columns as category codes, no scaling) and, for every
instance, the three nearest other instances are found (Euclidean distance, as in imbalanced-learn's default).
An instance counts as tied when its 3rd and 4th nearest other instances are at exactly the same distance, so that the
set of three neighbours is not unique, and as duplicated when another instance has exactly the same feature values.
Output: tie_check_f1_K5.csv (dataset, instances, tied_pct, duplicated_pct)
"""
import os, sys, logging, warnings, importlib.util
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_order  # noqa: F401

warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
SPLIT = os.environ.get("SPLIT_SCRIPT", "../src/original/federated_data_split_path2.py")
DATA = os.environ.get("DATA_DIR", "../datasets_original/").rstrip("/") + "/"
spec = importlib.util.spec_from_file_location("p2", SPLIT)
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
FOLDER = {"RT": "rt-iot2022", "LR": "letter+recognition", "UNSW": "unsw-nb15", "SLS": "statlog+landsat+satellite",
          "CAR": "car+evaluation", "SVS": "statlog+vehicle+silhouettes", "DRY": "dry+bean+dataset",
          "CM": "crowdsourced+mapping", "AIH": "aviaere_influenza_fallliste"}
I, K = 1, 5
rows = []
for code in (sys.argv[1:] or list(FOLDER)):
    df, name = p2.load_dataset(DATA + FOLDER[code]); lab = p2.detect_label_column(df, dataset_name=name)
    folds = p2.stratified_kfold_split(df, lab, 5, 42)
    Ditr = pd.concat([folds[j] for j in range(5) if j != I - 1], ignore_index=True)
    subsets = p2.stratified_split(Ditr, lab, K, 42 + I * 10 + K)
    n = tied = dup = 0
    for c in subsets:
        X = c.drop(columns=[lab]).copy()
        for col in X.select_dtypes(include=["object", "str", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes
        X = X.to_numpy(dtype=float)
        d, _ = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)
        # column 0 is the instance itself (or an exact duplicate of it); columns 1..3 are the three neighbours
        n += len(X); tied += int(np.sum(d[:, 3] == d[:, 4])); dup += int(np.sum(d[:, 1] == 0))
    rows.append(dict(dataset=code, instances=n, tied_pct=round(100 * tied / n, 2), duplicated_pct=round(100 * dup / n, 2)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("tie_check_f1_K5.csv", index=False)
