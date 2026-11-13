#!/usr/bin/env python3
"""Re-score the runs in which undersampling removed a class from every client of a federation.

fl_training.encode_y fits the label encoder on the clients' training labels only and maps every validation or test
label it has not seen to code 0. When a class is absent from all clients (with ENN on AIH in 14 federations and on RT in
one; see audit_classes_orig.csv), the validation and test instances of that class are scored as instances of the
alphabetically first class and the class drops out of the macro averages.

For these federations the script reads the original client, validation and test files (the files written by the split
scripts on the machine that ran the experiment), trains the local models with the original functions and seeds and scores
the aggregated predictions twice:
  orig_*  as in fl_training.py; equal to the stored values when the re-trained models are the same as the original ones;
  corr_*  with the labels coded over all classes of the training, validation and test sets, so that the removed class
          counts in the macro averages (it is never predicted, so its F1 is 0) and its instances count as errors.
The models and their predictions are not changed; only the scoring is.

Environment: FED_DIR (root of the staged data_splits folder), RAW (fl_results.csv), SRC_DIR (fl_training.py).
Outputs: rescored_vanished_runs.csv (one row per run), rescored_vanished_clients.csv (local models on their own).
"""
import os, sys, tempfile, importlib.util, logging, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# absolute paths: the script imports fl_training.py from a temporary working directory (see below)
FED_DIR = os.path.abspath(os.environ.get("FED_DIR", "../results/vanished_federations"))
RAW = os.path.abspath(os.environ.get("RAW", "../results/fl_results_raw.csv"))
SRC_DIR = os.path.abspath(os.environ.get("SRC_DIR", "../src/original"))
FOLDER = {"AIH": "Avian_Influenza_H5N1(AIH)", "RT": "RT-IoT2022(RT)", "CAR": "Car_Evaluation(CAR)",
          "SVS": "Statlog_Vehicle_Silhouettes(SVS)", "SLS": "Statlog_Landsat_Satellite(SLS)", "CM": "Crowdsourced_Mapping(CM)",
          "DRY": "Dry_Bean(DRY)", "LR": "Letter_Recognition(LR)", "UNSW": "UNSW_NB15(UNSW)"}
LEARNERS = ["DT", "RF", "GB", "XGB"]
RULES = ["FedAvg", "WeightedFedAvg", "FedMedian", "FedTrimmedMean", "FedProx", "FedBoost"]
METRICS = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "precision_weighted", "recall_macro",
           "recall_weighted", "auc_roc"]

_cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())          # fl_training.py opens a log file on import
spec = importlib.util.spec_from_file_location("flt", f"{SRC_DIR}/fl_training.py")
flt = importlib.util.module_from_spec(spec); spec.loader.exec_module(flt)
os.chdir(_cwd)
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")

# federations in which a class present before undersampling is absent from every client afterwards
a = pd.read_csv("audit_classes_orig.csv", dtype={"cls": str})
f = a.groupby(["dataset", "fold", "K", "method", "cls"]).agg(nb=("n_before", "sum"), na=("n_after", "sum")).reset_index()
fed = f[(f.nb > 0) & (f.na == 0)].groupby(["dataset", "fold", "K", "method"]).cls.apply(list).reset_index()
print(len(fed), "federations:", fed.groupby(["dataset", "method"]).size().to_dict())

raw = pd.read_csv(RAW)
raw = raw[raw.path_type == "us_after"]


def corrected(y_true_full, y_pred, proba, n_full):
    labels = list(range(n_full))
    p = np.zeros((proba.shape[0], n_full)); p[:, :proba.shape[1]] = proba
    m = {"accuracy": accuracy_score(y_true_full, y_pred)}
    for avg in ("macro", "weighted"):
        m[f"f1_{avg}"] = f1_score(y_true_full, y_pred, labels=labels, average=avg, zero_division=0)
        m[f"precision_{avg}"] = precision_score(y_true_full, y_pred, labels=labels, average=avg, zero_division=0)
        m[f"recall_{avg}"] = recall_score(y_true_full, y_pred, labels=labels, average=avg, zero_division=0)
    try:
        m["auc_roc"] = (roc_auc_score(y_true_full, p[:, 1]) if n_full == 2 else
                        roc_auc_score(y_true_full, p, multi_class="ovr", average="macro", labels=labels))
    except Exception:
        m["auc_roc"] = np.nan
    return m


runs, local = [], []
for _, fr in fed.iterrows():
    code, i, K, meth = fr.dataset, int(fr.fold), int(fr.K), fr.method
    d = f"{FED_DIR}/undersampling_after_split/{meth}/{FOLDER[code]}/f{code}_fold_{i}/{K}_federation"
    df_val = pd.read_csv(f"{d}/f{code}_fold_{i}_{K}_federation_val.csv")
    df_test = pd.read_csv(f"{d}/f{code}_fold_{i}_{K}_federation_test.csv")
    client_dfs = [pd.read_csv(f"{d}/f{code}_fold_{i}_{K}_federation_tr_{t}_{meth}.csv") for t in range(1, K + 1)]
    # --- as in fl_training.run_experiment ---
    label_col = flt.detect_label_column(df_val)
    all_train = pd.concat(client_dfs, ignore_index=True)
    X_all = all_train.drop(columns=[label_col])
    X_all_proc, X_val_proc = flt.preprocess(X_all, df_val.drop(columns=[label_col]))
    _, X_test_proc = flt.preprocess(X_all, df_test.drop(columns=[label_col]))
    y_all_enc, y_val_enc, y_test_enc, le = flt.encode_y(all_train[label_col], df_val[label_col], df_test[label_col])
    n_classes = len(le.classes_)
    client_data, offset = [], 0
    for cdf in client_dfs:
        n = len(cdf); client_data.append((X_all_proc[offset:offset + n], y_all_enc[offset:offset + n])); offset += n
    kept_idx = [t for t, (Xc, yc) in enumerate(client_data) if len(Xc) > 0 and len(np.unique(yc)) >= 2]
    # --- corrected coding: the classes seen in training keep their codes, unseen classes are appended ---
    seen = [str(c) for c in le.classes_]
    unseen = sorted((set(df_val[label_col].astype(str)) | set(df_test[label_col].astype(str))) - set(seen))
    full = {c: k for k, c in enumerate(seen + unseen)}
    yv_full = df_val[label_col].astype(str).map(full).to_numpy()
    yt_full = df_test[label_col].astype(str).map(full).to_numpy()
    assert set(unseen) == set(fr.cls), (code, i, K, unseen, fr.cls)
    for learner in LEARNERS:
        val_probs, test_probs, sizes, n_cls = flt.train_clients_once(client_data, X_val_proc, X_test_proc, learner,
                                                                      global_n_classes=n_classes)
        for pos, t in enumerate(kept_idx):
            yk = np.argmax(test_probs[pos], axis=1)
            local.append(dict(dataset=code, fold=i, K=K, us_method=meth, local_model=learner, client=t + 1,
                              f1_local_orig=f1_score(y_test_enc, yk, average="macro", zero_division=0),
                              f1_local_corr=f1_score(yt_full, yk, labels=list(range(len(full))), average="macro",
                                                     zero_division=0)))
        for rule in RULES:
            av = flt.aggregate_fl(val_probs, rule, sizes); at = flt.aggregate_fl(test_probs, rule, sizes)
            pv, pt = np.argmax(av, axis=1), np.argmax(at, axis=1)
            mo = {"val": flt.compute_metrics(y_val_enc, pv, av, n_cls), "test": flt.compute_metrics(y_test_enc, pt, at, n_cls)}
            mc = {"val": corrected(yv_full, pv, av, len(full)), "test": corrected(yt_full, pt, at, len(full))}
            st = raw[(raw.us_method == meth) & (raw.dataset == code) & (raw.fold == i) & (raw.n_clients == K)
                     & (raw.fl_algorithm == rule) & (raw.local_model == learner)]
            assert len(st) == 1, (code, i, K, rule, learner, len(st))
            row = dict(dataset=code, fold=i, K=K, us_method=meth, fl_algorithm=rule, local_model=learner,
                       removed_class=";".join(unseen), n_test=len(yt_full),
                       n_test_removed=int(df_test[label_col].astype(str).isin(unseen).sum()))
            for part in ("val", "test"):
                for m in METRICS:
                    row[f"{part}_{m}_stored"] = st[f"{part}_{m}"].iloc[0]
                    row[f"{part}_{m}_orig"] = round(float(mo[part][m]), 4)
                    row[f"{part}_{m}_corr"] = round(float(mc[part][m]), 4)
            runs.append(row)
    print(code, i, K, meth, "removed:", unseen, flush=True)

r = pd.DataFrame(runs); r.to_csv("rescored_vanished_runs.csv", index=False)
pd.DataFrame(local).to_csv("rescored_vanished_clients.csv", index=False)
same = np.isclose(r.test_f1_macro_orig, r.test_f1_macro_stored, atol=5e-5)
print(len(r), "runs; original scoring equals the stored F1-macro in", int(same.sum()), "runs")
print(r.assign(same=same).groupby(["dataset", "local_model"]).same.mean())
print("mean test F1-macro stored", round(100 * r.test_f1_macro_stored.mean(), 2), "corrected", round(100 * r.test_f1_macro_corr.mean(), 2))
