#!/usr/bin/env python3
"""
Skrypt uczenia federacyjnego (FL) na wygenerowanych plikach CSV.

Protokół: JEDNORUNDOWY (one-shot). Każdy klient trenuje swój model lokalny jeden raz
i przekazuje serwerowi wektory prawdopodobieństw klas dla przykładów walidacyjnych i testowych.
Serwer łączy je jedną z reguł poniżej i wybiera klasę o największym prawdopodobieństwie.
Nie ma rund komunikacji ani odsyłania modelu globalnego do klientów; 5-krotna walidacja
krzyżowa powtarza ten protokół na pięciu podziałach danych i nie dodaje rund.

Reguły agregacji (identyfikator w kodzie i w wynikach -> nazwa w artykule):
    FedAvg          -> FedAvg (średnia jednostajna prawdopodobieństw)
    WeightedFedAvg  -> weighted FedAvg (średnia ważona liczebnością danych klienta po US)
    FedMedian       -> coordinate-wise median
    FedTrimmedMean  -> trimmed mean
    FedProx         -> distance-weighted mean (wagi 1 / (1 + mu * ||P_k - P_mean||_F));
                       NIE jest to iteracyjny algorytm FedProx, tylko reguła łączenia wyjść
    FedBoost        -> rank-weighted mean (waga klienta k = 2k / (K(K + 1)));
                       NIE jest to boosting ani algorytm wielorundowy
    FedBagging      -> identyczny z FedAvg (w artykule połączony z FedAvg)
Identyfikatory pozostawiono bez zmian, aby zgadzały się z opublikowanymi wynikami.

Modele lokalne: Decision Tree (DT), Random Forest (RF), Gradient Boosting (GB), XGBoost (XGB)

Ścieżki:
    - baseline                    (bez undersamplingu)
    - undersampling_before_split  (US przed podziałem na klientów)
    - undersampling_after_split   (US po podziale, na każdym kliencie)

Metryki: Accuracy, F1 macro, F1 weighted, Precision, Recall,
         AUC-ROC (OvR), Confusion Matrix, czas trenowania

Wyniki zapisywane do:
    results/fl_results.csv       — wszystkie kombinacje w jednym pliku
    results/fl_results.xlsx      — to samo w Excelu z formatowaniem
    results/confusion_matrices/  — macierze pomyłek (JSON)
"""

import os
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fl_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# SYSTEM CHECKPOINTÓW
# ============================================
class CheckpointManager:
    """
    Zarządza stanem eksperymentów — zapisuje postęp i wznawia od miejsca przerwania.

    Pliki:
        results/checkpoint.json  — lista ukończonych eksperymentów (klucze)
        results/fl_results.csv   — wyniki (dopisywane na bieżąco)

    Klucz: "path_type|us_method|dataset|fold|n_clients|fl_alg|model"
    """

    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.checkpoint_file = results_path / "checkpoint.json"
        self.csv_file = results_path / "fl_results.csv"
        results_path.mkdir(parents=True, exist_ok=True)
        self.completed = self._load_checkpoint()

    def _load_checkpoint(self) -> set:
        completed = set()
        # 1. Wczytaj z pliku checkpoint.json (jeśli istnieje)
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            completed = set(data.get("completed", []))

        # 2. ŹRÓDŁO PRAWDY: odtwórz klucze z samego CSV (eliminuje duplikaty po restartach)
        if self.csv_file.exists():
            try:
                df = pd.read_csv(self.csv_file)
                for r in df.itertuples(index=False):
                    key = (f"{r.path_type}|{r.us_method}|{r.dataset}|"
                           f"{r.fold}|{r.n_clients}|{r.fl_algorithm}|{r.local_model}")
                    completed.add(key)
            except Exception as e:
                logger.warning(f"Nie udało się odczytać kluczy z CSV: {e}")

        if completed:
            logger.info(f"Checkpoint wczytany — {len(completed)} ukończonych eksperymentów "
                        f"(z CSV + checkpoint.json)")
        else:
            logger.info("Brak checkpointu — zaczynam od początku")
        return completed

    def _save_checkpoint(self):
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "completed": list(self.completed),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(self.completed)
            }, f, ensure_ascii=False, indent=2)

    def make_key(self, path_type, us_method, dataset, fold, n_clients, fl_alg, model) -> str:
        return f"{path_type}|{us_method or 'none'}|{dataset}|{fold}|{n_clients}|{fl_alg}|{model}"

    def is_done(self, key: str) -> bool:
        return key in self.completed

    def mark_done(self, key: str, row: dict):
        """Oznacza eksperyment jako ukończony i dopisuje wynik do CSV."""
        self.completed.add(key)
        df_row = pd.DataFrame([row])
        write_header = not self.csv_file.exists()
        df_row.to_csv(self.csv_file, mode='a', header=write_header,
                      index=False, encoding='utf-8-sig')
        # Checkpoint co 10 eksperymentów
        if len(self.completed) % 10 == 0:
            self._save_checkpoint()

    def finalize(self):
        self._save_checkpoint()
        logger.info(f"Checkpoint zapisany — łącznie: {len(self.completed)} eksperymentów")

    def progress(self) -> str:
        return f"Postęp: {len(self.completed)} ukończonych"


# ============================================
# KONFIGURACJA — ZMIEŃ TUTAJ
# ============================================
BASE_PATH = r"C:\Users\veronika\Desktop\data"

SPLIT_PATHS = {
    "baseline":  "data_splits",
    "us_before": "data_splits/undersampling_before_split",
    "us_after":  "data_splits/undersampling_after_split",
}

RESULTS_PATH = "results"

N_FOLDS      = 5
MIN_CLIENTS  = 3
MAX_CLIENTS  = 10
RANDOM_SEED  = 42
FEDPROX_MU   = 0.01   # mu w regule ważonej odległością ("FedProx"); jednorundowa reguła łączenia, nie trening FedProx
TRIMMED_FRAC = 0.1    # % skrajnych wartości do odcięcia w FedTrimmedMean
# (usunięto nieużywaną stałą FL_ROUNDS: protokół jest jednorundowy, żadna reguła nie używa rund)

DATASETS = {
    "RT":   "RT-IoT2022(RT)",
    "LR":   "Letter_Recognition(LR)",
    "UNSW": "UNSW_NB15(UNSW)",
    "SLS":  "Statlog_Landsat_Satellite(SLS)",
    "SOY":  "Soybean_Large(SOY)",
    "CAR":  "Car_Evaluation(CAR)",
    "SVS":  "Statlog_Vehicle_Silhouettes(SVS)",
    "DRY":  "Dry_Bean(DRY)",
    "CM":   "Crowdsourced_Mapping(CM)",
    "AIH":  "Avian_Influenza_H5N1(AIH)",
}

US_METHODS = [
    "RUS", "RUS25", "RUS50", "RUS75",
    "NearMiss", "NM3",
    "ENN", "AllKNN", "NCL",
    "TomekLinks",
]

FL_ALGORITHMS = [
    "FedAvg", "WeightedFedAvg", "FedMedian",
    "FedTrimmedMean", "FedProx", "FedBagging", "FedBoost"
]

LOCAL_MODELS = ["RF", "GB", "DT", "XGB"]

# Filtry CLI (uzupełniane w main())
FOLDS_FILTER = None
CLIENTS_FILTER = None
# ============================================


# ============================================
# MODELE LOKALNE
# ============================================
def get_model(model_name: str, random_state: int = RANDOM_SEED):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    elif model_name == "GB":
        return GradientBoostingClassifier(
            n_estimators=100, random_state=random_state, learning_rate=0.1
        )
    elif model_name == "DT":
        return DecisionTreeClassifier(
            random_state=random_state, max_depth=10
        )
    elif model_name == "XGB":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost nie jest zainstalowany. Uruchom: pip install xgboost")
        return XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        )
    raise ValueError(f"Nieznany model: {model_name}")


# ============================================
# PREPROCESSING
# ============================================
def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Label encoding kolumn kategorycznych. Fit na train, transform na test."""
    X_tr = X_train.copy().reset_index(drop=True)
    X_te = X_test.copy().reset_index(drop=True)

    for col in X_tr.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        known = set(le.classes_)
        X_te[col] = X_te[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    # Wypełnij NaN medianą
    X_tr = X_tr.fillna(X_tr.median(numeric_only=True))
    X_te = X_te.fillna(X_tr.median(numeric_only=True))

    return X_tr.values, X_te.values


def encode_y(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
    """Label coding over all classes of the training, validation and test labels.

    Fixed: the original version fitted the encoder on the clients' training labels only and mapped every validation or
    test label it had not seen to code 0. When undersampling removed a class from every client, the validation and test
    instances of that class were therefore scored as the first class and the class dropped out of the macro averages.
    Coding over all classes keeps the relative order of the training classes, so the trained models and their
    predictions are unchanged; a class that no client holds receives probability 0 and counts in the evaluation."""
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_val, y_test]).astype(str))
    return (le.transform(y_train.astype(str)), le.transform(y_val.astype(str)),
            le.transform(y_test.astype(str)), le)


# ============================================
# METRYKI
# ============================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, n_classes: int) -> dict:
    """Oblicza wszystkie metryki ewaluacji."""
    metrics = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average='macro',    zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision_macro":   precision_score(y_true, y_pred, average='macro',    zero_division=0),
        "precision_weighted":precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_macro":      recall_score(y_true, y_pred, average='macro',    zero_division=0),
        "recall_weighted":   recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # AUC-ROC
    try:
        if n_classes == 2:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
    except Exception:
        metrics["auc_roc"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


# ============================================
# ALGORYTMY AGREGACJI FL
# ============================================
def aggregate_probs(client_probs: list, method: str,
                    client_sizes: list = None) -> np.ndarray:
    """
    Agreguje prawdopodobieństwa klientów metodą FL.
    client_probs: lista tablic (n_samples, n_classes)
    """
    stacked = np.stack(client_probs, axis=0)  # (n_clients, n_samples, n_classes)

    if method == "FedAvg":
        return np.mean(stacked, axis=0)

    elif method == "WeightedFedAvg":
        if client_sizes is None:
            client_sizes = [1] * len(client_probs)
        total = sum(client_sizes)
        weights = np.array([s / total for s in client_sizes])
        return np.einsum('k,kij->ij', weights, stacked)

    elif method == "FedMedian":
        return np.median(stacked, axis=0)

    elif method == "FedTrimmedMean":
        n = len(client_probs)
        k = max(1, int(n * TRIMMED_FRAC))
        sorted_stack = np.sort(stacked, axis=0)
        trimmed = sorted_stack[k: n - k] if n - 2 * k > 0 else stacked
        return np.mean(trimmed, axis=0)

    elif method == "FedProx":
        # FedProx: ważona średnia z regularyzacją (dla predykcji: jak FedAvg
        # ale z korektą wag przez odległość od średniej globalnej)
        global_mean = np.mean(stacked, axis=0)
        weights = []
        for prob in client_probs:
            dist = np.linalg.norm(prob - global_mean)
            weights.append(1.0 / (1.0 + FEDPROX_MU * dist))
        total = sum(weights)
        weights = [w / total for w in weights]
        return np.einsum('k,kij->ij', np.array(weights), stacked)

    else:
        return np.mean(stacked, axis=0)


def train_clients(client_data: list, label_column: str,
                  X_val: np.ndarray, model_name: str,
                  fl_algorithm: str, random_state: int = RANDOM_SEED):
    """
    Trenuje modele na danych klientów i zwraca zagregowane prawdopodobieństwa
    na zbiorze walidacyjnym.

    Obsługuje: FedAvg, WeightedFedAvg, FedMedian, FedTrimmedMean, FedProx,
               FedBagging, FedBoost
    """
    client_probs = []
    client_sizes = []
    trained_models = []

    # --- Trenowanie lokalnych modeli ---
    for t, (X_c, y_c) in enumerate(client_data):
        if len(X_c) == 0 or len(np.unique(y_c)) < 2:
            logger.warning(f"  Klient {t+1}: za mało danych lub 1 klasa — pomijam")
            continue

        model = get_model(model_name, random_state + t)
        model.fit(X_c, y_c)
        trained_models.append(model)
        client_sizes.append(len(X_c))

        proba = model.predict_proba(X_val)
        client_probs.append(proba)

    if not client_probs:
        raise ValueError("Żaden klient nie dostarczył danych do trenowania")

    # Wyrównaj liczbę klas (jeśli różne klasy na różnych klientach)
    n_classes = max(p.shape[1] for p in client_probs)
    aligned_probs = []
    for model, prob in zip(trained_models, client_probs):
        if prob.shape[1] < n_classes:
            aligned = np.zeros((prob.shape[0], n_classes))
            for i, cls_idx in enumerate(model.classes_):
                if cls_idx < n_classes:
                    aligned[:, cls_idx] = prob[:, i]
            aligned_probs.append(aligned)
        else:
            aligned_probs.append(prob)

    # --- Agregacja według algorytmu FL ---
    if fl_algorithm == "FedBagging":
        # Voting: każdy model głosuje, agregacja przez średnią predykcji
        all_preds = np.stack(
            [m.predict(X_val) for m in trained_models], axis=0
        )
        # Soft voting przez uśrednienie prawdopodobieństw
        agg_probs = np.mean(np.stack(aligned_probs, axis=0), axis=0)

    elif fl_algorithm == "FedBoost":
        # Sekwencyjny boosting: każdy kolejny model uczy się na błędach poprzedniego
        # Implementacja uproszczona: ważona agregacja gdzie późniejsze modele mają większą wagę
        n = len(aligned_probs)
        boost_weights = np.array([(i + 1) for i in range(n)], dtype=float)
        boost_weights /= boost_weights.sum()
        agg_probs = np.einsum('k,kij->ij', boost_weights, np.stack(aligned_probs, axis=0))

    else:
        # FedAvg, WeightedFedAvg, FedMedian, FedTrimmedMean, FedProx
        agg_probs = aggregate_probs(aligned_probs, fl_algorithm, client_sizes)

    return agg_probs, trained_models, n_classes


# ============================================
# OPTYMALIZACJA: trenuj modele RAZ, agreguj wielokrotnie
# ============================================
def train_clients_once(client_data: list, X_val: np.ndarray, X_test: np.ndarray,
                       model_name: str, global_n_classes: int = None,
                       random_state: int = RANDOM_SEED):
    """
    Trenuje lokalne modele klientów JEDEN RAZ i zwraca wyrównane (po klasach)
    prawdopodobieństwa na zbiorach val i test — osobno dla każdego klienta.

    Te prawdopodobieństwa są następnie agregowane przez WSZYSTKIE algorytmy FL
    bez ponownego trenowania modeli (modele lokalne nie zależą od metody agregacji).

    Zwraca: (val_probs_aligned, test_probs_aligned, client_sizes, n_classes)
    """
    val_probs, test_probs, client_sizes, trained_models = [], [], [], []
    client_classes = []  # oryginalne (globalne) etykiety klas obecne u klienta

    for t, (X_c, y_c) in enumerate(client_data):
        if len(X_c) == 0 or len(np.unique(y_c)) < 2:
            logger.warning(f"  Klient {t+1}: za mało danych lub 1 klasa — pomijam")
            continue
        model = get_model(model_name, random_state + t)
        if model_name == "XGB":
            # XGBoost wymaga ciągłych etykiet 0..k-1; remapuj lokalnie i zapamiętaj oryginały
            uniq = np.unique(y_c)
            remap = {c: i for i, c in enumerate(uniq)}
            y_c_fit = np.array([remap[v] for v in y_c])
            model.fit(X_c, y_c_fit)
            client_classes.append(uniq)
        else:
            model.fit(X_c, y_c)
            client_classes.append(np.asarray(model.classes_))
        trained_models.append(model)
        client_sizes.append(len(X_c))
        val_probs.append(model.predict_proba(X_val))
        test_probs.append(model.predict_proba(X_test))

    if not val_probs:
        raise ValueError("Żaden klient nie dostarczył danych do trenowania")

    # Liczba klas = globalna (z label encodera) lub max obserwowana u klientów
    n_classes = max(p.shape[1] for p in val_probs)
    max_cls_idx = max(int(c) for classes in client_classes for c in classes) + 1
    n_classes = max(n_classes, max_cls_idx)
    if global_n_classes is not None:
        n_classes = max(n_classes, global_n_classes)

    def _align(probs_list):
        out = []
        for classes, prob in zip(client_classes, probs_list):
            # Zawsze mapuj kolumny prob -> globalne indeksy klas (classes[i])
            aligned = np.zeros((prob.shape[0], n_classes))
            for i, cls_idx in enumerate(classes):
                if int(cls_idx) < n_classes:
                    aligned[:, int(cls_idx)] = prob[:, i]
            out.append(aligned)
        return out

    return _align(val_probs), _align(test_probs), client_sizes, n_classes


def aggregate_fl(aligned_probs: list, fl_algorithm: str, client_sizes: list) -> np.ndarray:
    """Stosuje wybraną metodę agregacji FL na gotowych (wyrównanych) prawdopodobieństwach."""
    if fl_algorithm == "FedBagging":
        return np.mean(np.stack(aligned_probs, axis=0), axis=0)
    elif fl_algorithm == "FedBoost":
        n = len(aligned_probs)
        bw = np.array([(i + 1) for i in range(n)], dtype=float)
        bw /= bw.sum()
        return np.einsum('k,kij->ij', bw, np.stack(aligned_probs, axis=0))
    else:
        # FedAvg, WeightedFedAvg, FedMedian, FedTrimmedMean, FedProx
        return aggregate_probs(aligned_probs, fl_algorithm, client_sizes)


# ============================================
# WCZYTYWANIE PLIKÓW CSV
# ============================================
def load_fold_data(base_path: Path, acronym: str, dataset_folder: str,
                   fold: int, n_clients: int,
                   path_type: str, us_method: str = None) -> dict:
    """
    Wczytuje pliki CSV dla danego foldu i federacji.
    Zwraca słownik z danymi val, test i listą danych klientów.
    """
    if path_type == "baseline":
        fold_path = base_path / dataset_folder / f"f{acronym}_fold_{fold}" / f"{n_clients}_federation"
        suffix = ""
    elif path_type == "us_before":
        fold_path = (base_path / us_method / dataset_folder /
                     f"f{acronym}_fold_{fold}" / f"{n_clients}_federation")
        suffix = f"_{us_method}"
    elif path_type == "us_after":
        fold_path = (base_path / us_method / dataset_folder /
                     f"f{acronym}_fold_{fold}" / f"{n_clients}_federation")
        suffix = ""  # val/test bez suffixu, tr z suffixem

    if not fold_path.exists():
        raise FileNotFoundError(f"Brak folderu: {fold_path}")

    # Val i test
    val_file  = fold_path / f"f{acronym}_fold_{fold}_{n_clients}_federation_val{suffix}.csv"
    test_file = fold_path / f"f{acronym}_fold_{fold}_{n_clients}_federation_test{suffix}.csv"

    # Fallback dla us_after (val/test bez suffixu)
    if path_type == "us_after":
        val_file  = fold_path / f"f{acronym}_fold_{fold}_{n_clients}_federation_val.csv"
        test_file = fold_path / f"f{acronym}_fold_{fold}_{n_clients}_federation_test.csv"

    df_val  = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    # Dane klientów
    client_dfs = []
    for t in range(1, n_clients + 1):
        tr_suffix = f"_{us_method}" if path_type in ["us_before", "us_after"] else ""
        tr_file = fold_path / f"f{acronym}_fold_{fold}_{n_clients}_federation_tr_{t}{tr_suffix}.csv"
        if tr_file.exists():
            client_dfs.append(pd.read_csv(tr_file))
        else:
            logger.warning(f"Brak pliku: {tr_file}")

    return {"val": df_val, "test": df_test, "clients": client_dfs}


def detect_label_column(df: pd.DataFrame) -> str:
    """Wykrywa kolumnę etykiet."""
    known = ['attack_type', 'lettr', 'label', 'class', 'Class', 'outcome',
             'target', 'y', 'diagnosis', 'category', 'type', 'species']
    cols_lower = {c.lower(): c for c in df.columns}
    for k in known:
        if k.lower() in cols_lower:
            return cols_lower[k.lower()]
    return df.columns[-1]


# ============================================
# GŁÓWNA FUNKCJA EKSPERYMENTU
# ============================================
def run_experiment(base_split_path: Path, results_path: Path,
                   path_type: str, us_method: str = None,
                   checkpoint: "CheckpointManager" = None):
    """
    Uruchamia eksperymenty FL dla wszystkich kombinacji:
    dataset × fold × n_clients × fl_algorithm × local_model

    Obsługuje wznowienie od miejsca przerwania przez CheckpointManager.
    path_type: "baseline" | "us_before" | "us_after"
    us_method: "RUS" | "NearMiss" | "TomekLinks" | "ENN" | None
    """
    tag = path_type if us_method is None else f"{path_type}_{us_method}"
    logger.info(f"\n{'='*70}")
    logger.info(f"EKSPERYMENT: {tag}")
    if checkpoint:
        logger.info(f"{checkpoint.progress()}")
    logger.info(f"{'='*70}")

    all_results = []
    cm_path = results_path / "confusion_matrices" / tag
    cm_path.mkdir(parents=True, exist_ok=True)

    for acronym, dataset_folder in DATASETS.items():
        logger.info(f"\n--- Dataset: {acronym} ({dataset_folder}) ---")

        for fold in range(1, N_FOLDS + 1):
            if FOLDS_FILTER is not None and fold not in FOLDS_FILTER:
                continue
            for n_clients in range(MIN_CLIENTS, MAX_CLIENTS + 1):
                if CLIENTS_FILTER is not None and n_clients not in CLIENTS_FILTER:
                    continue

                # Wczytaj dane (raz dla wszystkich kombinacji FL × model)
                # load_fold_data sam dokleja us_method dla us_before/us_after,
                # więc przekazujemy base_split_path bez us_method (unikamy podwójnej nazwy)
                try:
                    data = load_fold_data(
                        base_split_path, acronym, dataset_folder,
                        fold, n_clients, path_type, us_method
                    )
                except FileNotFoundError as e:
                    logger.warning(f"  Pominięto fold={fold}, j={n_clients}: {e}")
                    continue

                df_val     = data["val"]
                df_test    = data["test"]
                client_dfs = data["clients"]

                if not client_dfs:
                    continue

                # Sprawdź czy WSZYSTKIE kombinacje tego foldu są już gotowe
                all_done = checkpoint and all(
                    checkpoint.is_done(
                        checkpoint.make_key(path_type, us_method, acronym,
                                            fold, n_clients, fl_alg, model_name)
                    )
                    for fl_alg in FL_ALGORITHMS
                    for model_name in LOCAL_MODELS
                )
                if all_done:
                    logger.info(f"   Pominięto (już gotowe): fold={fold}, j={n_clients}")
                    continue

                # Preprocessing (raz na fold × n_clients)
                label_col  = detect_label_column(df_val)
                all_train  = pd.concat(client_dfs, ignore_index=True)
                X_all      = all_train.drop(columns=[label_col])
                X_val_raw  = df_val.drop(columns=[label_col])
                X_test_raw = df_test.drop(columns=[label_col])
                y_val_raw  = df_val[label_col]
                y_test_raw = df_test[label_col]

                X_all_proc,  X_val_proc  = preprocess(X_all, X_val_raw)
                _,           X_test_proc = preprocess(X_all, X_test_raw)

                y_all_raw = all_train[label_col]
                y_all_enc, y_val_enc, y_test_enc, le = encode_y(
                    y_all_raw, y_val_raw, y_test_raw
                )
                n_classes = len(le.classes_)

                client_data = []
                offset = 0
                for cdf in client_dfs:
                    n = len(cdf)
                    X_c = X_all_proc[offset:offset + n]
                    y_c = y_all_enc[offset:offset + n]
                    client_data.append((X_c, y_c))
                    offset += n

                # === OPTYMALIZACJA: trenuj modele RAZ na model, agreguj 7× ===
                for model_name in LOCAL_MODELS:

                    # Pomiń trenowanie jeśli WSZYSTKIE algorytmy FL dla tego modelu gotowe
                    if checkpoint and all(
                        checkpoint.is_done(checkpoint.make_key(
                            path_type, us_method, acronym, fold, n_clients, fl_alg, model_name))
                        for fl_alg in FL_ALGORITHMS
                    ):
                        logger.info(f"   Pomijam (checkpoint): wszystkie FL dla {model_name}")
                        continue

                    # Trenuj lokalne modele klientów JEDEN RAZ
                    try:
                        t_train_start = time.time()
                        val_probs, test_probs, client_sizes, n_cls = train_clients_once(
                            client_data, X_val_proc, X_test_proc, model_name,
                            global_n_classes=n_classes
                        )
                        train_time = time.time() - t_train_start
                    except Exception as e:
                        logger.error(f"  BŁĄD trenowania {tag}|{acronym}|fold={fold}|"
                                     f"j={n_clients}|{model_name} -> {e}")
                        continue

                    # Agreguj przez wszystkie algorytmy FL (bez ponownego trenowania)
                    for fl_alg in FL_ALGORITHMS:
                        key = checkpoint.make_key(
                            path_type, us_method, acronym,
                            fold, n_clients, fl_alg, model_name
                        ) if checkpoint else None

                        if checkpoint and checkpoint.is_done(key):
                            continue

                        exp_id = (f"{tag} | {acronym} | fold={fold} | "
                                  f"j={n_clients} | {fl_alg} | {model_name}")
                        logger.info(f"  -> {exp_id}")

                        try:
                            t_agg_start = time.time()
                            agg_probs_val  = aggregate_fl(val_probs,  fl_alg, client_sizes)
                            agg_probs_test = aggregate_fl(test_probs, fl_alg, client_sizes)
                            y_pred_val  = np.argmax(agg_probs_val,  axis=1)
                            y_pred_test = np.argmax(agg_probs_test, axis=1)
                            agg_time = time.time() - t_agg_start

                            val_metrics  = compute_metrics(y_val_enc,  y_pred_val,
                                                           agg_probs_val,  n_cls)
                            test_metrics = compute_metrics(y_test_enc, y_pred_test,
                                                           agg_probs_test, n_cls)

                            # Confusion matrix
                            cm_file = cm_path / f"{acronym}_fold{fold}_j{n_clients}_{fl_alg}_{model_name}.json"
                            with open(cm_file, 'w') as f:
                                json.dump({
                                    "val_cm":  val_metrics["confusion_matrix"],
                                    "test_cm": test_metrics["confusion_matrix"],
                                    "classes": list(le.classes_)
                                }, f)

                            row = {
                                "path_type":             path_type,
                                "us_method":             us_method or "none",
                                "dataset":               acronym,
                                "fold":                  fold,
                                "n_clients":             n_clients,
                                "fl_algorithm":          fl_alg,
                                "local_model":           model_name,
                                "train_time_sec":        round(train_time + agg_time, 3),
                                "val_accuracy":          round(val_metrics["accuracy"],           4),
                                "val_f1_macro":          round(val_metrics["f1_macro"],           4),
                                "val_f1_weighted":       round(val_metrics["f1_weighted"],        4),
                                "val_precision_macro":   round(val_metrics["precision_macro"],    4),
                                "val_precision_weighted":round(val_metrics["precision_weighted"], 4),
                                "val_recall_macro":      round(val_metrics["recall_macro"],       4),
                                "val_recall_weighted":   round(val_metrics["recall_weighted"],    4),
                                "val_auc_roc":           round(val_metrics["auc_roc"], 4) if not np.isnan(val_metrics["auc_roc"]) else None,
                                "test_accuracy":         round(test_metrics["accuracy"],           4),
                                "test_f1_macro":         round(test_metrics["f1_macro"],           4),
                                "test_f1_weighted":      round(test_metrics["f1_weighted"],        4),
                                "test_precision_macro":  round(test_metrics["precision_macro"],    4),
                                "test_precision_weighted":round(test_metrics["precision_weighted"],4),
                                "test_recall_macro":     round(test_metrics["recall_macro"],       4),
                                "test_recall_weighted":  round(test_metrics["recall_weighted"],    4),
                                "test_auc_roc":          round(test_metrics["auc_roc"], 4) if not np.isnan(test_metrics["auc_roc"]) else None,
                            }

                            all_results.append(row)

                            if checkpoint:
                                checkpoint.mark_done(key, row)

                        except Exception as e:
                            logger.error(f"  BŁĄD: {exp_id} -> {e}")
                            import traceback
                            traceback.print_exc()

    return all_results

    for acronym, dataset_folder in DATASETS.items():
        logger.info(f"\n--- Dataset: {acronym} ({dataset_folder}) ---")

        for fold in range(1, N_FOLDS + 1):
            for n_clients in range(MIN_CLIENTS, MAX_CLIENTS + 1):

                # Wczytaj dane
                try:
                    split_base = base_split_path / (us_method or "")
                    if path_type == "baseline":
                        split_base = base_split_path
                    data = load_fold_data(
                        split_base, acronym, dataset_folder,
                        fold, n_clients, path_type, us_method
                    )
                except FileNotFoundError as e:
                    logger.warning(f"  Pominięto fold={fold}, j={n_clients}: {e}")
                    continue

                df_val  = data["val"]
                df_test = data["test"]
                client_dfs = data["clients"]

                if not client_dfs:
                    continue

                # Wykryj etykietę
                label_col = detect_label_column(df_val)

                # Przygotuj dane globalne do preprocessingu
                all_train = pd.concat(client_dfs, ignore_index=True)
                X_all = all_train.drop(columns=[label_col])

                X_val_raw  = df_val.drop(columns=[label_col])
                X_test_raw = df_test.drop(columns=[label_col])
                y_val_raw  = df_val[label_col]
                y_test_raw = df_test[label_col]

                # Preprocessing (fit na wszystkich danych treningowych)
                X_all_proc, X_val_proc = preprocess(X_all, X_val_raw)
                _,          X_test_proc = preprocess(X_all, X_test_raw)

                y_all_raw = all_train[label_col]
                y_all_enc, y_val_enc, y_test_enc, le = encode_y(
                    y_all_raw, y_val_raw, y_test_raw
                )
                n_classes = len(le.classes_)

                # Przygotuj dane klientów (zakodowane)
                client_data = []
                offset = 0
                for cdf in client_dfs:
                    n = len(cdf)
                    X_c = X_all_proc[offset:offset + n]
                    y_c = y_all_enc[offset:offset + n]
                    client_data.append((X_c, y_c))
                    offset += n

                # Eksperymenty dla każdej kombinacji FL × model
                for fl_alg in FL_ALGORITHMS:
                    for model_name in LOCAL_MODELS:

                        exp_id = (f"{tag} | {acronym} | fold={fold} | "
                                  f"j={n_clients} | {fl_alg} | {model_name}")
                        logger.info(f"  {exp_id}")

                        try:
                            t_start = time.time()

                            # Trenuj klientów i agreguj na val
                            agg_probs_val, trained_models, n_cls = train_clients(
                                client_data, label_col,
                                X_val_proc, model_name, fl_alg
                            )

                            # Predykcja na val (wybór klasy z max prawdopodobieństwa)
                            y_pred_val = np.argmax(agg_probs_val, axis=1)

                            # Agregacja na test
                            test_probs_list = []
                            for model in trained_models:
                                p = model.predict_proba(X_test_proc)
                                if p.shape[1] < n_cls:
                                    aligned = np.zeros((p.shape[0], n_cls))
                                    for i, ci in enumerate(model.classes_):
                                        if ci < n_cls:
                                            aligned[:, ci] = p[:, i]
                                    test_probs_list.append(aligned)
                                else:
                                    test_probs_list.append(p)

                            if fl_alg == "FedBagging":
                                agg_probs_test = np.mean(np.stack(test_probs_list), axis=0)
                            elif fl_alg == "FedBoost":
                                n = len(test_probs_list)
                                bw = np.array([(i+1) for i in range(n)], dtype=float)
                                bw /= bw.sum()
                                agg_probs_test = np.einsum(
                                    'k,kij->ij', bw, np.stack(test_probs_list))
                            else:
                                client_sizes = [len(cd[0]) for cd in client_data]
                                agg_probs_test = aggregate_probs(
                                    test_probs_list, fl_alg, client_sizes)

                            y_pred_test = np.argmax(agg_probs_test, axis=1)
                            t_end = time.time()

                            # Metryki
                            val_metrics  = compute_metrics(y_val_enc,  y_pred_val,  agg_probs_val,  n_cls)
                            test_metrics = compute_metrics(y_test_enc, y_pred_test, agg_probs_test, n_cls)

                            # Zapisz confusion matrix
                            cm_file = cm_path / f"{acronym}_fold{fold}_j{n_clients}_{fl_alg}_{model_name}.json"
                            with open(cm_file, 'w') as f:
                                json.dump({
                                    "val_cm":  val_metrics["confusion_matrix"],
                                    "test_cm": test_metrics["confusion_matrix"],
                                    "classes": list(le.classes_)
                                }, f)

                            # Zapisz wynik
                            row = {
                                "path_type":        path_type,
                                "us_method":        us_method or "none",
                                "dataset":          acronym,
                                "fold":             fold,
                                "n_clients":        n_clients,
                                "fl_algorithm":     fl_alg,
                                "local_model":      model_name,
                                "train_time_sec":   round(t_end - t_start, 3),
                                # Val metryki
                                "val_accuracy":         round(val_metrics["accuracy"],          4),
                                "val_f1_macro":         round(val_metrics["f1_macro"],          4),
                                "val_f1_weighted":      round(val_metrics["f1_weighted"],       4),
                                "val_precision_macro":  round(val_metrics["precision_macro"],   4),
                                "val_precision_weighted":round(val_metrics["precision_weighted"],4),
                                "val_recall_macro":     round(val_metrics["recall_macro"],      4),
                                "val_recall_weighted":  round(val_metrics["recall_weighted"],   4),
                                "val_auc_roc":          round(val_metrics["auc_roc"],           4) if not np.isnan(val_metrics["auc_roc"]) else None,
                                # Test metryki
                                "test_accuracy":         round(test_metrics["accuracy"],          4),
                                "test_f1_macro":         round(test_metrics["f1_macro"],          4),
                                "test_f1_weighted":      round(test_metrics["f1_weighted"],       4),
                                "test_precision_macro":  round(test_metrics["precision_macro"],   4),
                                "test_precision_weighted":round(test_metrics["precision_weighted"],4),
                                "test_recall_macro":     round(test_metrics["recall_macro"],      4),
                                "test_recall_weighted":  round(test_metrics["recall_weighted"],   4),
                                "test_auc_roc":          round(test_metrics["auc_roc"],           4) if not np.isnan(test_metrics["auc_roc"]) else None,
                            }
                            all_results.append(row)

                        except Exception as e:
                            logger.error(f"  BŁĄD: {exp_id} -> {e}")
                            import traceback
                            traceback.print_exc()

    return all_results


# ============================================
# ZAPIS WYNIKÓW
# ============================================
def save_results(all_results: list, results_path: Path):
    """Zapisuje wyniki do CSV i Excel."""
    if not all_results:
        logger.warning("Brak wyników do zapisania!")
        return

    df = pd.DataFrame(all_results)
    results_path.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_file = results_path / "fl_results.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"Zapisano CSV: {csv_file}")

    # Excel z formatowaniem
    xlsx_file = results_path / "fl_results.xlsx"
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Wszystkie wyniki', index=False)

        # Arkusz ze średnimi per dataset/metoda
        summary = df.groupby(
            ['path_type', 'us_method', 'dataset', 'fl_algorithm', 'local_model']
        ).agg(
            avg_test_accuracy=('test_accuracy', 'mean'),
            avg_test_f1_macro=('test_f1_macro', 'mean'),
            avg_test_f1_weighted=('test_f1_weighted', 'mean'),
            avg_test_auc_roc=('test_auc_roc', 'mean'),
            avg_train_time=('train_time_sec', 'mean'),
            n_experiments=('test_accuracy', 'count'),
        ).reset_index()
        summary = summary.round(4)
        summary.to_excel(writer, sheet_name='Podsumowanie', index=False)

        # Formatowanie
        wb = writer.book
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    logger.info(f"Zapisano Excel: {xlsx_file}")
    logger.info(f"Łącznie wyników: {len(df)}")

    # Podsumowanie w konsoli
    logger.info("\n=== TOP 10 kombinacji (test F1 macro) ===")
    top = df.nlargest(10, 'test_f1_macro')[
        ['dataset', 'path_type', 'us_method', 'fl_algorithm',
         'local_model', 'n_clients', 'test_f1_macro', 'test_accuracy']
    ]
    logger.info(f"\n{top.to_string(index=False)}")


# ============================================
# CLI — uruchamianie w partiach
# ============================================
def parse_args():
    p = argparse.ArgumentParser(
        description="FL training na wygenerowanych splitach — można uruchamiać partiami (checkpoint).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady (uruchamianie w partiach):

  # Cały baseline (10 datasetów, 7 FL, 4 modele)
  python fl_training.py --paths baseline

  # US before split — tylko 4 metody RUS
  python fl_training.py --paths us_before --us RUS RUS25 RUS50 RUS75

  # US after split — tylko cleaning methods
  python fl_training.py --paths us_after --us ENN AllKNN NCL

  # Wybrany dataset + jeden model (szybki test)
  python fl_training.py --paths baseline --datasets AIH --models DT

  # Wznowienie wszystkiego (checkpoint pomija to co już zrobione)
  python fl_training.py --paths baseline us_before us_after
        """
    )
    p.add_argument("--paths", nargs="+", default=["baseline", "us_before", "us_after"],
                   choices=["baseline", "us_before", "us_after"],
                   help="Które ścieżki uruchomić (domyślnie wszystkie).")
    p.add_argument("--us", nargs="+", default=None,
                   help=f"Lista metod US (domyślnie wszystkie z US_METHODS). Dostępne: {US_METHODS}")
    p.add_argument("--datasets", nargs="+", default=None,
                   help=f"Lista akronimów datasetów (domyślnie wszystkie). Dostępne: {list(DATASETS.keys())}")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="Lista foldów (1-5, domyślnie wszystkie).")
    p.add_argument("--clients", nargs="+", type=int, default=None,
                   help="Lista liczby klientów (3-10, domyślnie wszystkie).")
    p.add_argument("--fl", nargs="+", default=None,
                   help=f"Lista algorytmów FL (domyślnie wszystkie). Dostępne: {FL_ALGORITHMS}")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Lista modeli lokalnych (domyślnie wszystkie). Dostępne: {LOCAL_MODELS}")
    p.add_argument("--results", default=RESULTS_PATH,
                   help=f"Folder wyników (domyślnie {RESULTS_PATH}).")
    return p.parse_args()


# ============================================
# MAIN
# ============================================
def main():
    args = parse_args()
    base = Path(BASE_PATH)
    results_path = base / args.results

    # --- Filtruj listy globalne wg argumentów CLI ---
    global DATASETS, FL_ALGORITHMS, LOCAL_MODELS, US_METHODS
    global N_FOLDS, MIN_CLIENTS, MAX_CLIENTS, FOLDS_FILTER, CLIENTS_FILTER
    if args.datasets:
        DATASETS = {k: v for k, v in DATASETS.items() if k in args.datasets}
    if args.fl:
        FL_ALGORITHMS = [a for a in FL_ALGORITHMS if a in args.fl]
    if args.models:
        LOCAL_MODELS = [m for m in LOCAL_MODELS if m in args.models]
    selected_us = args.us if args.us else US_METHODS

    if args.folds:
        FOLDS_FILTER = set(args.folds)
        N_FOLDS = max(args.folds)
    if args.clients:
        CLIENTS_FILTER = set(args.clients)
        MIN_CLIENTS = min(args.clients)
        MAX_CLIENTS = max(args.clients)

    logger.info(f"Konfiguracja:")
    logger.info(f"  Ścieżki:   {args.paths}")
    logger.info(f"  Datasety:  {list(DATASETS.keys())}")
    logger.info(f"  US:        {selected_us}")
    logger.info(f"  Foldy:     {args.folds or list(range(1, N_FOLDS+1))}")
    logger.info(f"  Klienci:   {MIN_CLIENTS}-{MAX_CLIENTS}")
    logger.info(f"  FL:        {FL_ALGORITHMS}")
    logger.info(f"  Modele:    {LOCAL_MODELS}")
    logger.info(f"  Wyniki:    {results_path}")

    # === CHECKPOINT — jeden dla całego przebiegu ===
    checkpoint = CheckpointManager(results_path)
    logger.info(f"Stan startowy: {checkpoint.progress()}")

    try:
        if "baseline" in args.paths:
            logger.info("\n" + "#"*70)
            logger.info("ŚCIEŻKA 3 — BASELINE (bez undersamplingu)")
            logger.info("#"*70)
            baseline_path = base / SPLIT_PATHS["baseline"]
            run_experiment(baseline_path, results_path, "baseline", None, checkpoint)

        if "us_before" in args.paths:
            logger.info("\n" + "#"*70)
            logger.info("ŚCIEŻKA 1 — UNDERSAMPLING PRZED PODZIAŁEM")
            logger.info("#"*70)
            us_before_path = base / SPLIT_PATHS["us_before"]
            for us_method in selected_us:
                logger.info(f"\n>>> Metoda US: {us_method}")
                run_experiment(us_before_path, results_path, "us_before", us_method, checkpoint)

        if "us_after" in args.paths:
            logger.info("\n" + "#"*70)
            logger.info("ŚCIEŻKA 2 — UNDERSAMPLING PO PODZIALE (na klientach)")
            logger.info("#"*70)
            us_after_path = base / SPLIT_PATHS["us_after"]
            for us_method in selected_us:
                logger.info(f"\n>>> Metoda US: {us_method}")
                run_experiment(us_after_path, results_path, "us_after", us_method, checkpoint)

    except KeyboardInterrupt:
        logger.info("\n[UWAGA]  Przerwano przez użytkownika (Ctrl+C)")
        logger.info("Postęp zapisany — uruchom ponownie aby kontynuować")
    finally:
        # Zawsze zapisz checkpoint na końcu (nawet po błędzie/przerwaniu)
        checkpoint.finalize()

        # Wygeneruj Excel ze wszystkich wyników w CSV
        if (results_path / "fl_results.csv").exists():
            save_results(pd.read_csv(results_path / "fl_results.csv").to_dict('records'),
                         results_path)

    logger.info("\n" + "="*70)
    logger.info(f"ZAKOŃCZONO — {checkpoint.progress()}")
    logger.info(f"Wyniki: {results_path / 'fl_results.xlsx'}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
