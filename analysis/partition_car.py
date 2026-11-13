#!/usr/bin/env python3
"""Class counts of the sets produced by stages 1 to 3 for Car Evaluation, fold 1 (input of the partition figure).
Uses the original baseline split script and seeds. Output: partition_car_f1.csv"""
import os, sys, logging, warnings, importlib.util
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_order  # noqa: F401

warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
SPLIT = os.environ.get("SPLIT_BASE", "../src/original/federated_data_split.py")
DATA = os.environ.get("DATA_DIR", "../datasets_original/").rstrip("/") + "/"
spec = importlib.util.spec_from_file_location("base", SPLIT)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
df, name = b.load_dataset(DATA + "car+evaluation")
lab = b.detect_label_column(df, dataset_name=name); df[lab] = df[lab].astype(str)
folds = b.stratified_kfold_split(df, lab, 5, 42)
i = 1
Di = folds[i - 1]; Ditr = pd.concat([folds[j] for j in range(5) if j != i - 1], ignore_index=True)
val, test = b.stratified_val_test_split(Di, lab, 42 + i)
rows = []
def add(group, label, d):
    vc = d[lab].value_counts()
    rows.append(dict(group=group, label=label, **{c: int(vc.get(c, 0)) for c in ["unacc", "acc", "good", "vgood"]}))
add("stage 1", "Data set D", df)
add("stage 2, fold 1", "Training part", Ditr)
add("stage 2, fold 1", "Validation half", val)
add("stage 2, fold 1", "Test half", test)
for K, shown in [(3, [1, 2, 3]), (10, [1, 10])]:
    subsets = b.stratified_split(Ditr, lab, K, 42 + i * 10 + K)
    for t in shown:
        add(f"stage 3, K = {K}", f"Client {t} of {K}", subsets[t - 1])
pd.DataFrame(rows).to_csv("partition_car_f1.csv", index=False)
print(pd.DataFrame(rows).to_string())
