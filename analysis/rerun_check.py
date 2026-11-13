#!/usr/bin/env python3
"""Reproduction check and per-client view of stages 6 and 7.

Re-creates the client sets of one fold and one federation size with the original split scripts and seeds,
trains the local models with the original training functions (fl_training.py) and records
  1. the test metrics of every aggregation rule, next to the values stored in the raw results;
  2. the test F1-macro of every local model on its own (argmax of its own probability vectors);
  3. how often the six aggregation rules predict the same class.
Nothing here enters the 86 400 analysed runs; the script only re-computes some of them.

Environment variables: SRC_DIR (original scripts), DATA_DIR (raw data sets), RAW (fl_results.csv),
FOLD (default 1), K (default 5), DATASETS (comma-separated codes, default all nine).
Outputs: rerun_runs_f{FOLD}_K{K}.csv, rerun_clients_f{FOLD}_K{K}.csv, rerun_agreement_f{FOLD}_K{K}.csv
"""
import os, io, sys, time, importlib.util, logging, warnings, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import win_order  # noqa: F401  (file order as on Windows, where the experiment ran)
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = Path.cwd()
# defaults: the folders of the package, relative to this script; absolute because the original scripts are imported
# from a temporary working directory (see below)
_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_DIR = os.path.abspath(os.environ.get("SRC_DIR", os.path.join(_PKG, "src", "original")))
DATA_DIR = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(_PKG, "datasets_original")))
RAW = os.path.abspath(os.environ.get("RAW", os.path.join(_PKG, "results", "fl_results_raw.csv")))
FOLD = int(os.environ.get("FOLD", 1))
K = int(os.environ.get("K", 5))
FOLDER = {"AIH": "aviaere_influenza_fallliste", "SVS": "statlog+vehicle+silhouettes", "CAR": "car+evaluation",
          "SLS": "statlog+landsat+satellite", "CM": "crowdsourced+mapping", "DRY": "dry+bean+dataset",
          "LR": "letter+recognition", "UNSW": "unsw-nb15", "RT": "rt-iot2022"}
CODES = os.environ.get("DATASETS", ",".join(FOLDER)).split(",")
METHODS = ["RUS", "RUS25", "RUS50", "RUS75", "NearMiss", "NM3", "ENN", "AllKNN", "TomekLinks"]
LEARNERS = ["DT", "RF", "GB", "XGB"]
RULES = ["FedAvg", "WeightedFedAvg", "FedMedian", "FedTrimmedMean", "FedProx", "FedBoost"]
SEED = 42


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# fl_training.py opens a log file in the working directory on import: import it from a temporary directory
_cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())
base = load_module("split_base", f"{SRC_DIR}/federated_data_split.py")
p2 = load_module("split_p2", f"{SRC_DIR}/federated_data_split_path2.py")
flt = load_module("flt", f"{SRC_DIR}/fl_training.py")
os.chdir(_cwd)
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")

raw = pd.read_csv(RAW)
raw = raw[(raw.fold == FOLD) & (raw.n_clients == K) & raw.path_type.isin(["baseline", "us_after"])]


def csv_roundtrip(d):
    """The original pipeline wrote every set to CSV and read it back; do the same so dtypes match."""
    buf = io.StringIO(); d.to_csv(buf, index=False); buf.seek(0)
    return pd.read_csv(buf)


def federation_sets(code):
    """Return {condition: (val, test, [client sets])} exactly as the split scripts wrote them."""
    path = f"{DATA_DIR}/{FOLDER[code]}"
    sets = {}
    # baseline: federated_data_split.py
    df, name = base.load_dataset(path)
    lab = base.detect_label_column(df, dataset_name=name)
    df[lab] = df[lab].astype(str)
    folds = base.stratified_kfold_split(df, lab, 5, SEED)
    Di = folds[FOLD - 1]
    Ditr = pd.concat([folds[j] for j in range(5) if j != FOLD - 1], ignore_index=True)
    val, test = base.stratified_val_test_split(Di, lab, SEED + FOLD)
    subsets = base.stratified_split(Ditr, lab, K, SEED + FOLD * 10 + K)
    sets["none"] = (val, test, subsets)
    # per-client undersampling: federated_data_split_path2.py
    df2, name2 = p2.load_dataset(path)
    lab2 = p2.detect_label_column(df2, dataset_name=name2)
    folds2 = p2.stratified_kfold_split(df2, lab2, 5, SEED)
    Di2 = folds2[FOLD - 1]
    Ditr2 = pd.concat([folds2[j] for j in range(5) if j != FOLD - 1], ignore_index=True)
    val2, test2 = p2.stratified_val_test_split(Di2, lab2, SEED + FOLD)
    subsets2 = p2.stratified_split(Ditr2, lab2, K, SEED + FOLD * 10 + K)
    for m in METHODS:
        clients = [p2.apply_undersampling(subsets2[t - 1], lab2, m, random_state=SEED + FOLD * 100 + K * 10 + t)
                   for t in range(1, K + 1)]
        sets[m] = (val2, test2, clients)
    return sets


runs, clients_rows, agree_rows = [], [], []
for code in CODES:
    t_ds = time.time()
    sets = federation_sets(code)
    for cond, (val, test, clients) in sets.items():
        df_val, df_test = csv_roundtrip(val), csv_roundtrip(test)
        client_dfs = [csv_roundtrip(c) for c in clients]
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
        for learner in LEARNERS:
            t0 = time.time()
            val_probs, test_probs, sizes, n_cls = flt.train_clients_once(client_data, X_val_proc, X_test_proc, learner,
                                                                          global_n_classes=n_classes)
            t_train = time.time() - t0
            # local models on their own
            for pos, t in enumerate(kept_idx):
                yk = np.argmax(test_probs[pos], axis=1)
                clients_rows.append(dict(dataset=code, fold=FOLD, K=K, us_method=cond, local_model=learner, client=t + 1,
                                         n_train=sizes[pos], n_classes_train=len(np.unique(client_data[t][1])),
                                         f1_local=f1_score(y_test_enc, yk, average="macro", zero_division=0)))
            preds = {}
            for rule in RULES:
                agg = flt.aggregate_fl(test_probs, rule, sizes)
                yp = np.argmax(agg, axis=1); preds[rule] = yp
                met = flt.compute_metrics(y_test_enc, yp, agg, n_cls)
                path_type = "baseline" if cond == "none" else "us_after"
                st = raw[(raw.path_type == path_type) & (raw.us_method == cond) & (raw.dataset == code)
                         & (raw.fl_algorithm == rule) & (raw.local_model == learner)]
                row = dict(dataset=code, fold=FOLD, K=K, us_method=cond, fl_algorithm=rule, local_model=learner,
                           n_clients_trained=len(kept_idx), train_time_rerun=round(t_train, 3))
                for mname in ("f1_macro", "accuracy", "precision_macro", "recall_macro"):
                    row[f"test_{mname}_rerun"] = round(met[mname], 4)
                    row[f"test_{mname}_stored"] = st[f"test_{mname}"].iloc[0] if len(st) == 1 else np.nan
                row["n_stored_rows"] = len(st)
                runs.append(row)
            P = np.vstack([preds[r] for r in RULES])
            all_same = (P == P[0]).all(axis=0).mean()
            ar = dict(dataset=code, fold=FOLD, K=K, us_method=cond, local_model=learner, n_test=P.shape[1],
                      share_all_rules_agree=round(all_same, 6))
            for r in RULES[1:]:
                ar[f"agree_{r}_vs_FedAvg"] = round((preds[r] == preds["FedAvg"]).mean(), 6)
            agree_rows.append(ar)
    pd.DataFrame(runs).to_csv(OUT / f"rerun_runs_f{FOLD}_K{K}.csv", index=False)
    pd.DataFrame(clients_rows).to_csv(OUT / f"rerun_clients_f{FOLD}_K{K}.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(OUT / f"rerun_agreement_f{FOLD}_K{K}.csv", index=False)
    r = pd.DataFrame(runs); r = r[r.dataset == code]
    same = (r.test_f1_macro_rerun == r.test_f1_macro_stored).mean()
    print(f"{code}: {len(r)} runs, identical F1 in {same:.1%}, max |diff| "
          f"{(r.test_f1_macro_rerun - r.test_f1_macro_stored).abs().max():.4f}, {time.time() - t_ds:.0f} s", flush=True)
