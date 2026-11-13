#!/usr/bin/env python3
"""
ŚCIEŻKA 2 — Podział na klientów FL, a NASTĘPNIE undersampling na każdym kliencie osobno.

Kolejność operacji:
    1. Wczytaj cały zbiór D
    2. 5CV: podziel D na 5 foldów
    3. Dla każdego foldu i:
       - Di_val (50%), Di_test (50%), Ditr = D \\ Di
    4. Dla każdej federacji j (3–10):
       - Podziel Ditr na j podzbiorów (klientów): Ditr,1 … Ditr,j
       - Na KAŻDYM podzbiorze Ditr,t zastosuj undersampling -> Ditr,t_us
       - Zapisz pliki CSV

Metody undersamplingu: RUS, NearMiss, TomekLinks, ENN
Struktura wyjściowa:
    data_splits/undersampling_after_split/{METODA}/{Dataset(AKRONIM)}/...
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# KONFIGURACJA - ZMIEŃ TUTAJ
# ============================================
N_FOLDS = 5
MIN_CLIENTS = 3
MAX_CLIENTS = 10
RANDOM_SEED = 42
BASE_PATH = r"C:\Users\veronika\Desktop\data"

ALL_METHODS = ["RUS", "RUS25", "RUS50", "RUS75",
               "NearMiss", "NM3",
               "ENN", "AllKNN", "NCL",
               "TomekLinks"]
# Domyślnie generuj tylko brakujące. Zmień na konkretne metody aby ograniczyć.
UNDERSAMPLING_METHODS = ALL_METHODS

DATASETS = [
    ("datasets_original/rt-iot2022",                  "data_splits/undersampling_after_split",      "RT"),
    ("datasets_original/letter+recognition",          "data_splits/undersampling_after_split",      "LR"),
    ("datasets_original/unsw-nb15",                   "data_splits/undersampling_after_split",      "UNSW"),
    ("datasets_original/statlog+landsat+satellite",   "data_splits/undersampling_after_split",      "SLS"),
    ("datasets_original/soybean+large",               "data_splits/undersampling_after_split",      "SOY"),
    ("datasets_original/car+evaluation",              "data_splits/undersampling_after_split",      "CAR"),
    ("datasets_original/statlog+vehicle+silhouettes", "data_splits/undersampling_after_split",      "SVS"),
    ("datasets_original/dry+bean+dataset",            "data_splits/undersampling_after_split",      "DRY"),
    ("datasets_original/crowdsourced+mapping",        "data_splits/undersampling_after_split",      "CM"),
    ("datasets_original/aviaere_influenza_fallliste", "data_splits/undersampling_after_split",      "AIH"),
]

DATASET_OUTPUT_NAMES = {
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
# ============================================

# ============================================
# KONFIGURACJA DATASETÓW
# ============================================
DATASET_CONFIG = {
    "rt-iot2022": {
        "label": "Attack_type", "header": True, "task": "classification",
        "id": "RT", "drop_columns": ["Unnamed: 0"],
    },
    "letter+recognition": {
        "label": "lettr", "header": False, "task": "classification", "id": "LR",
        "columns": ["lettr","x-box","y-box","width","high","onpix","x-bar","y-bar",
                    "x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"],
    },
    "unsw-nb15": {
        "label": "label", "header": True, "task": "classification",
        "id": "UNSW", "drop_columns": ["id", "attack_cat"],
    },
    "UNSW_NB15_training-set": {
        "label": "label", "header": True, "task": "classification",
        "id": "UNSW", "drop_columns": ["id", "attack_cat"],
    },
    "UNSW_NB15_testing-set": {
        "label": "label", "header": True, "task": "classification",
        "id": "UNSW", "drop_columns": ["id", "attack_cat"],
    },
    "statlog+landsat+satellite": {
        "label": 36, "header": False, "task": "classification", "sep": " ", "id": "SLS",
    },
    "sat": {
        "label": 36, "header": False, "task": "classification", "sep": " ", "id": "SLS",
    },
    "soybean+large": {
        "label": 0, "header": False, "task": "classification", "id": "SOY",
    },
    "soybean-large": {
        "label": 0, "header": False, "task": "classification", "id": "SOY",
    },
    "car+evaluation": {
        "label": "class", "header": False, "task": "classification", "id": "CAR",
        "columns": ["buying","maint","doors","persons","lug_boot","safety","class"],
    },
    "statlog+vehicle+silhouettes": {
        "label": 18, "header": False, "task": "classification", "sep": " ", "id": "SVS",
    },
    "dry+bean+dataset": {
        "label": "Class", "header": True, "task": "classification", "id": "DRY",
    },
    "dry_bean_dataset": {
        "label": "Class", "header": True, "task": "classification", "id": "DRY",
    },
    "crowdsourced+mapping": {
        "label": "class", "header": True, "task": "classification", "id": "CM",
    },
    "avian+influenza+h5n1": {
        "label": "outcome", "header": 1, "task": "classification",
        "id": "AIH", "drop_columns": ["id"],
    },
    "aviaere_influenza_fallliste": {
        "label": "outcome", "header": 1, "task": "classification",
        "id": "AIH", "drop_columns": ["id"],
    },
}

KNOWN_LABEL_NAMES = [
    'target', 'label', 'class', 'y', 'output', 'outcome',
    'diagnosis', 'attack_type', 'category', 'type',
    'species', 'variety', 'action', 'result',
    'klasa', 'etykieta', 'cel', 'diabetes_012', 'diabetes_binary',
    'disposition', 'status', 'Class', 'lettr',
]
# ============================================


# ============================================
# FUNKCJE POMOCNICZE
# ============================================
def get_dataset_config(name: str) -> dict:
    if name in DATASET_CONFIG:
        return DATASET_CONFIG[name]
    name_no_ext = Path(name).stem
    if name_no_ext in DATASET_CONFIG:
        return DATASET_CONFIG[name_no_ext]
    name_lower = name.lower()
    for key, config in DATASET_CONFIG.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return config
    return None


def _read_file(file_path: Path, config: dict = None) -> pd.DataFrame:
    sep = ','
    header = 'infer'
    names = None
    if config:
        sep = config.get('sep', ',')
        cfg_header = config.get('header')
        if cfg_header is False:
            header = None
            names = config.get('columns', None)
        elif isinstance(cfg_header, int) and cfg_header is not True:
            header = cfg_header
        elif cfg_header is True:
            header = 0
    suffix = file_path.suffix.lower()
    if suffix in ['.csv', '.data']:
        return pd.read_csv(file_path, sep=sep, header=header, names=names)
    elif suffix in ['.trn', '.tst']:
        return pd.read_csv(file_path, sep=r'\s+', engine='python', header=header, names=names)
    elif suffix == '.dat':
        return pd.read_csv(file_path, sep=r'\s+', engine='python', header=header, names=names)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, header=header, names=names)
    elif suffix == '.txt':
        if sep != ',':
            return pd.read_csv(file_path, sep=sep, header=header, names=names)
        return pd.read_csv(file_path, sep=None, engine='python', header=header, names=names)
    else:
        return pd.read_csv(file_path, sep=sep, header=header, names=names)


def load_dataset(data_path: str) -> tuple:
    path = Path(data_path)
    config = get_dataset_config(path.name)

    IGNORE_FILES_LOWER = ['.gitignore', '.ds_store', 'index', 'license', 'readme',
                          'sha256sums', 'checksums', 'manifest', 'license.txt',
                          'readme.txt', 'sha256sums.txt', 'checksums.txt',
                          'vehicle.doc', 'sat.doc', 'soybean-explanation',
                          'stepp-order', 'fisher-order', 'why-various-soybean-databases']

    if path.is_file():
        df = _read_file(path, config)
        name = path.stem
    elif path.is_dir():
        data_files = (list(path.glob('**/*.csv')) + list(path.glob('**/*.data')) +
                      list(path.glob('**/*.dat')) + list(path.glob('**/*.trn')) +
                      list(path.glob('**/*.tst')) + list(path.glob('**/*.xlsx')) +
                      list(path.glob('**/*.xls')))
        txt_files = [f for f in path.glob('**/*.txt') if f.name.lower() not in IGNORE_FILES_LOWER]
        data_files += txt_files
        # Pliki bez rozszerzenia (np. RT_IOT2022)
        for f in path.rglob('*'):
            if f.is_file() and f.suffix == '' and f.name.lower() not in IGNORE_FILES_LOWER:
                data_files.append(f)
        data_files = [f for f in data_files if f.name.lower() not in IGNORE_FILES_LOWER]

        if not data_files:
            raise ValueError(f"Brak plików danych w: {path}")

        dfs = []
        for f in data_files:
            try:
                dfs.append(_read_file(f, config))
                logger.info(f"  Wczytano: {f.name} ({len(dfs[-1])} wierszy)")
            except Exception as e:
                logger.warning(f"  Pominięto {f.name}: {e}")

        df = pd.concat(dfs, ignore_index=True)
        name = path.name
    else:
        raise ValueError(f"Ścieżka nie istnieje: {path}")

    if config and 'drop_columns' in config:
        cols = [c for c in config['drop_columns'] if c in df.columns]
        if cols:
            df = df.drop(columns=cols)

    df = df.replace('?', np.nan).dropna()

    # Ujednolicenie typow w kolumnie etykiet
    if config and 'label' in config:
        label = config['label']
        if isinstance(label, int) and label < len(df.columns):
            label_col = df.columns[label]
        elif isinstance(label, str) and label in df.columns:
            label_col = label
        else:
            label_col = None
        if label_col:
            df[label_col] = df[label_col].astype(str)

    logger.info(f"Dataset '{name}': {len(df)} probek, {len(df.columns)} kolumn")
    return df, name


def detect_label_column(df: pd.DataFrame, dataset_name: str = None) -> str:
    if dataset_name:
        config = get_dataset_config(dataset_name)
        if config and 'label' in config:
            label = config['label']
            if isinstance(label, str) and label in df.columns:
                return label
            elif isinstance(label, int) and label < len(df.columns):
                return df.columns[label]
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for known in KNOWN_LABEL_NAMES:
        if known.lower() in df_cols_lower:
            return df_cols_lower[known.lower()]
    return df.columns[-1]


def apply_undersampling(df: pd.DataFrame, label_column: str, method: str,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Stosuje undersampling na pojedynczym podzbiorze (kliencie).
    Fallback do oryginału jeśli zbyt mało próbek lub błąd.
    """
    from imblearn.under_sampling import (
        RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours,
        AllKNN, NeighbourhoodCleaningRule
    )

    # Sprawdź czy undersampling ma sens (min. 2 klasy, min. 4 próbki)
    n_classes = df[label_column].nunique()
    if n_classes < 2 or len(df) < 4:
        logger.warning(f"    [{method}] Za mało danych ({len(df)} próbek, {n_classes} klas) — pomijam US")
        return df

    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Kodowanie kategoryczne
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=['object', 'str', 'category']).columns:
        X_enc[col] = X_enc[col].astype('category').cat.codes

    def _rus_ratio_strategy(y_series, ratio):
        counts = pd.Series(y_series).value_counts()
        majority = counts.idxmax()
        target = {}
        for cls, cnt in counts.items():
            target[cls] = max(1, int(cnt * ratio)) if cls == majority else int(cnt)
        return target

    try:
        if method == "RUS":
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == "RUS25":
            sampler = RandomUnderSampler(
                sampling_strategy=_rus_ratio_strategy(y, 0.25),
                random_state=random_state)
        elif method == "RUS50":
            sampler = RandomUnderSampler(
                sampling_strategy=_rus_ratio_strategy(y, 0.50),
                random_state=random_state)
        elif method == "RUS75":
            sampler = RandomUnderSampler(
                sampling_strategy=_rus_ratio_strategy(y, 0.75),
                random_state=random_state)
        elif method in ("NearMiss", "NM1"):
            sampler = NearMiss(version=1, n_jobs=-1)
        elif method == "NM3":
            sampler = NearMiss(version=3, n_jobs=-1)
        elif method == "TomekLinks":
            sampler = TomekLinks(n_jobs=-1)
        elif method == "ENN":
            sampler = EditedNearestNeighbours(n_jobs=-1)
        elif method == "AllKNN":
            sampler = AllKNN(n_jobs=-1)
        elif method == "NCL":
            sampler = NeighbourhoodCleaningRule(n_jobs=-1)
        else:
            raise ValueError(f"Nieznana metoda: {method}")

        _, y_res = sampler.fit_resample(X_enc, y)

        if hasattr(sampler, 'sample_indices_'):
            df_res = df.iloc[sampler.sample_indices_].copy().reset_index(drop=True)
        else:
            X_res, _ = sampler.fit_resample(X_enc, y)
            df_res = pd.DataFrame(X_res, columns=X.columns)
            df_res[label_column] = y_res.values

        return df_res

    except Exception as e:
        logger.warning(f"    [{method}] Błąd US: {e} — zwracam oryginał")
        return df


def stratified_kfold_split(df, label_column, n_splits, random_state=42):
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [df.iloc[idx].copy() for _, idx in skf.split(df.drop(columns=[label_column]), df[label_column])]
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [df.iloc[idx].copy() for _, idx in kf.split(df)]


def stratified_split(df, label_column, n_splits, random_state=42):
    if n_splits == 1:
        return [df.copy()]
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [df.iloc[idx].copy() for _, idx in skf.split(df.drop(columns=[label_column]), df[label_column])]
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [df.iloc[idx].copy() for _, idx in kf.split(df)]


def stratified_val_test_split(df, label_column, random_state=42):
    try:
        idx_val, idx_test = train_test_split(
            df.index, test_size=0.5, stratify=df[label_column], random_state=random_state)
        return df.loc[idx_val].copy(), df.loc[idx_test].copy()
    except ValueError:
        mid = len(df) // 2
        df_s = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return df_s.iloc[:mid].copy(), df_s.iloc[mid:].copy()


# ============================================
# GŁÓWNA FUNKCJA ŚCIEŻKI 2
# ============================================
def create_path2_datasets(data_path: str, output_base_path: str,
                          acronym: str, method: str,
                          random_state: int = None):
    """
    Ścieżka 2: 5CV -> podział na j klientów -> undersampling na każdym kliencie osobno.

    Val i test NIE są poddawane undersamplingowi (służą do oceny).
    Undersampling stosowany tylko na zbiorach treningowych klientów.

    Struktura wyjściowa:
        {output_base_path}/{method}/{Dataset(ACRONYM)}/
            f{ACR}_fold_i/
                j_federation/
                    f{ACR}_fold_i_j_federation_val.csv          <- bez US
                    f{ACR}_fold_i_j_federation_test.csv         <- bez US
                    f{ACR}_fold_i_j_federation_tr_t_{method}.csv <- z US
    """
    if random_state is None:
        random_state = RANDOM_SEED

    # --- Wczytaj dane ---
    df, name = load_dataset(data_path)
    label_column = detect_label_column(df, dataset_name=name)
    fname = acronym
    dataset_out_name = DATASET_OUTPUT_NAMES.get(acronym, f"{name}({acronym})")

    logger.info(f"\n{'='*60}")
    logger.info(f"ŚCIEŻKA 2 | Metoda: {method} | Dataset: {fname}")
    logger.info(f"Etykieta: '{label_column}' | Próbki: {len(df)}")
    logger.info(f"{'='*60}")

    # --- Folder wyjściowy: undersampling_after_split/METHOD/Dataset(ACR)/ ---
    output_path = Path(output_base_path) / method / dataset_out_name
    output_path.mkdir(parents=True, exist_ok=True)

    # --- 5CV na oryginalnym D (bez US) ---
    folds = stratified_kfold_split(df, label_column, N_FOLDS, random_state)
    logger.info(f"5CV podział:")
    for i, fold in enumerate(folds, 1):
        logger.info(f"  D{i}: {len(fold)} próbek")

    # --- Przygotuj dane dla każdego foldu ---
    fold_data = {}
    for i in range(1, N_FOLDS + 1):
        Di = folds[i - 1]
        Ditr = pd.concat([folds[j] for j in range(N_FOLDS) if j != i - 1], ignore_index=True)
        Di_val, Di_test = stratified_val_test_split(Di, label_column, random_state + i)

        fold_path = output_path / f"f{fname}_fold_{i}"
        fold_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Fold {i}: |Ditr|={len(Ditr)}, |val|={len(Di_val)}, |test|={len(Di_test)}")
        fold_data[i] = {'Di_val': Di_val, 'Di_test': Di_test, 'Ditr': Ditr, 'fold_path': fold_path}

    # --- Tworzenie federacji z undersamplingiem na każdym kliencie ---
    for i in range(1, N_FOLDS + 1):
        Di_val   = fold_data[i]['Di_val']
        Di_test  = fold_data[i]['Di_test']
        Ditr     = fold_data[i]['Ditr']
        fold_path = fold_data[i]['fold_path']

        for j in range(MIN_CLIENTS, MAX_CLIENTS + 1):
            # Podział Ditr na j podzbiorów (przed US)
            Ditr_subsets = stratified_split(Ditr, label_column, j, random_state + i * 10 + j)

            fed_path = fold_path / f"{j}_federation"
            fed_path.mkdir(parents=True, exist_ok=True)

            # Val i test — BEZ undersamplingu (wspólne dla wszystkich klientów)
            Di_val.to_csv(fed_path / f"f{fname}_fold_{i}_{j}_federation_val.csv", index=False)
            Di_test.to_csv(fed_path / f"f{fname}_fold_{i}_{j}_federation_test.csv", index=False)

            # Każdy klient — z undersamplingiem
            for t in range(1, j + 1):
                client_data = Ditr_subsets[t - 1]
                client_us = apply_undersampling(
                    client_data, label_column, method,
                    random_state=random_state + i * 100 + j * 10 + t
                )
                client_us.to_csv(
                    fed_path / f"f{fname}_fold_{i}_{j}_federation_tr_{t}_{method}.csv",
                    index=False
                )
                logger.info(f"  Fold {i} | j={j} | klient {t}: {len(client_data)}->{len(client_us)} próbek [{method}]")

    logger.info(f"\nZAKOŃCZONO [{method}] {fname} -> {output_path}")
    return output_path


def main():
    base = Path(BASE_PATH)

    for idx, (input_rel, output_rel, acronym) in enumerate(DATASETS, 1):
        input_path  = base / input_rel
        output_path = base / output_rel

        logger.info(f"\n{'#'*60}")
        logger.info(f"DATASET {idx}/{len(DATASETS)}: {acronym}")
        logger.info(f"{'#'*60}")

        for method in UNDERSAMPLING_METHODS:
            dataset_out_name = DATASET_OUTPUT_NAMES.get(acronym, acronym)
            target = base / output_rel / method / dataset_out_name / f"f{acronym}_fold_5" / f"{MAX_CLIENTS}_federation"
            if target.exists() and any(target.iterdir()):
                logger.info(f"  Pomiń [{method}] {acronym} — już istnieje: {target}")
                continue
            try:
                create_path2_datasets(
                    str(input_path), str(output_path),
                    acronym=acronym, method=method
                )
            except Exception as e:
                logger.error(f"Błąd [{method}] {acronym}: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"\n{'='*60}")
    logger.info(f"ŚCIEŻKA 2 — ZAKOŃCZONO WSZYSTKIE DATASETY")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
