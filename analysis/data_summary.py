#!/usr/bin/env python3
"""Summary of the nine prepared data sets (stage 1), computed with the original loading code.

For every data set: instances, features (numeric and categorical as read), classes, largest and smallest class,
imbalance ratio, share of exactly duplicated rows and the class counts. Used for the data set table and figures.
Environment variables: SPLIT_SCRIPT (path2 split script), DATA_DIR. Output: data_summary.csv, class_counts.csv
"""
import os, sys, logging, warnings, importlib.util
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_order  # noqa: F401

warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
SPLIT = os.environ.get("SPLIT_SCRIPT", "../src/original/federated_data_split_path2.py")
DATA = os.environ.get("DATA_DIR", "../datasets_original/").rstrip("/") + "/"
spec = importlib.util.spec_from_file_location("p2", SPLIT)
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
FOLDER = {"UNSW": "unsw-nb15", "RT": "rt-iot2022", "LR": "letter+recognition", "DRY": "dry+bean+dataset",
          "CM": "crowdsourced+mapping", "SLS": "statlog+landsat+satellite", "CAR": "car+evaluation",
          "SVS": "statlog+vehicle+silhouettes", "AIH": "aviaere_influenza_fallliste"}
rows, counts = [], []
for code, folder in FOLDER.items():
    df, name = p2.load_dataset(DATA + folder)
    lab = p2.detect_label_column(df, dataset_name=name)
    X = df.drop(columns=[lab])
    cat = X.select_dtypes(include=["object", "str", "category"]).columns
    vc = df[lab].value_counts()
    rows.append(dict(code=code, label=lab, instances=len(df), features=X.shape[1], categorical=len(cat),
                     numeric=X.shape[1] - len(cat), classes=len(vc), largest=int(vc.max()), smallest=int(vc.min()),
                     IR=vc.max() / vc.min(), duplicates_pct=100 * df.duplicated().mean(),
                     categorical_names=";".join(map(str, cat))))
    for cls, n in vc.items():
        counts.append(dict(code=code, cls=cls, n=int(n), share_pct=100 * n / len(df)))
    print(code, len(df), X.shape[1], len(vc), round(vc.max() / vc.min(), 2), flush=True)
pd.DataFrame(rows).to_csv("data_summary.csv", index=False)
pd.DataFrame(counts).to_csv("class_counts.csv", index=False)
