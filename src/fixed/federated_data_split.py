#!/usr/bin/env python3
"""
Skrypt do tworzenia zbiorów danych dla uczenia federacyjnego.
Realizacja pseudokodu dla podziału danych z użyciem 5-krotnej kroswalidacji
i tworzenia federacji o różnej liczbie klientów (3-10).
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from pathlib import Path
import logging
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# KONFIGURACJA - ZMIEŃ TUTAJ
# ============================================
N_FOLDS = 5  # Liczba foldów kroswalidacji (np. 5, 10)
MIN_CLIENTS = 3   # Minimalna liczba klientów federacji
MAX_CLIENTS = 10  # Maksymalna liczba klientów federacji
RANDOM_SEED = 42  # Ziarno losowości (42 = powtarzalny podział, None = losowy za każdym razem)
USE_STRATIFICATION = True  # True = stratyfikacja (zachowuje proporcje klas), False = losowy podział
BASE_PATH = r"C:\Users\veronika\Desktop\data"

# Lista datasetów: (folder/plik wejściowy, folder wyjściowy, akronim)
# Źródło: Zestawy_danych.xlsx — 10 zbiorów danych, klienci FL: 3–10
DATASETS = [
    ("datasets_original/rt-iot2022",                  "data_splits/RT-IoT2022(RT)",                     "RT"),
    ("datasets_original/letter+recognition",          "data_splits/Letter_Recognition(LR)",             "LR"),
    ("datasets_original/unsw-nb15",                   "data_splits/UNSW_NB15(UNSW)",                    "UNSW"),
    ("datasets_original/statlog+landsat+satellite",   "data_splits/Statlog_Landsat_Satellite(SLS)",     "SLS"),
    ("datasets_original/soybean+large",               "data_splits/Soybean_Large(SOY)",                 "SOY"),
    ("datasets_original/car+evaluation",              "data_splits/Car_Evaluation(CAR)",                "CAR"),
    ("datasets_original/statlog+vehicle+silhouettes", "data_splits/Statlog_Vehicle_Silhouettes(SVS)",   "SVS"),
    ("datasets_original/dry+bean+dataset",            "data_splits/Dry_Bean(DRY)",                      "DRY"),
    ("datasets_original/crowdsourced+mapping",        "data_splits/Crowdsourced_Mapping(CM)",           "CM"),
    ("datasets_original/aviaere_influenza_fallliste", "data_splits/Avian_Influenza_H5N1(AIH)",          "AIH"),
]
# ============================================

# ============================================
# KONFIGURACJA DATASETÓW
# ============================================
# Mapowanie: nazwa_folderu -> {
#   "label": kolumna docelowa (lub indeks int jeśli brak nagłówków),
#   "header": True/False (czy plik ma nagłówki),
#   "task": "classification" / "regression",
#   "columns": opcjonalna lista nazw kolumn (gdy header=False),
#   "sep": opcjonalny separator (domyślnie ','),
#   "id": akronim datasetu (używany jako fname w nazwach folderów/plików)
# }
DATASET_CONFIG = {
    # 1. RT-IoT2022 — klasyfikacja ataków sieciowych IoT
    "rt-iot2022": {
        "label": "Attack_type",
        "header": True,
        "task": "classification",
        "id": "RT",
        "drop_columns": ["Unnamed: 0"],
    },
    # 2. Letter Recognition — rozpoznawanie liter (A–Z)
    "letter+recognition": {
        "label": "lettr",
        "header": False,
        "task": "classification",
        "id": "LR",
        "columns": [
            "lettr", "x-box", "y-box", "width", "high", "onpix",
            "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr",
            "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
        ],
    },
    # 3. UNSW-NB15 — wykrywanie intruzji sieciowych
    # Pliki: UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv (skrypt scali automatycznie)
    # Etykieta: label — binarna (0=normalny ruch, 1=atak)
    # 257673 próbek łącznie, 43 cechy; id i attack_cat usunięte
    "unsw-nb15": {
        "label": "label",
        "header": True,
        "task": "classification",
        "id": "UNSW",
        "drop_columns": ["id", "attack_cat"],
    },
    "UNSW_NB15_training-set": {
        "label": "label",
        "header": True,
        "task": "classification",
        "id": "UNSW",
        "drop_columns": ["id", "attack_cat"],
    },
    "UNSW_NB15_testing-set": {
        "label": "label",
        "header": True,
        "task": "classification",
        "id": "UNSW",
        "drop_columns": ["id", "attack_cat"],
    },
    # 4. Statlog (Landsat Satellite) — klasyfikacja pikseli satelitarnych
    # Pliki: sat.trn + sat.tst (skrypt scali automatycznie)
    # Etykieta: ostatnia kolumna (indeks 36) — 6 klas (1,2,3,4,5,7 — brak klasy 6)
    # 6435 próbek łącznie, 36 cech numerycznych, separator spacja, brak nagłówków
    "statlog+landsat+satellite": {
        "label": 36,
        "header": False,
        "task": "classification",
        "sep": " ",
        "id": "SLS",
    },
    "sat": {
        "label": 36,
        "header": False,
        "task": "classification",
        "sep": " ",
        "id": "SLS",
    },
    # 5. Soybean (Large) — diagnostyka chorób soi
    # Pliki: soybean-large.data + soybean-large.test (skrypt scali automatycznie)
    # Etykieta: kolumna 0 (pierwsza) — 19 klas chorób soi
    # 683 próbek łącznie, 35 cech kategorycznych, wartości '?' zastępowane NaN i usuwane
    # UWAGA: klasa 'herbicide-injury' ma tylko 8 próbek — może powodować problemy przy stratyfikacji
    "soybean+large": {
        "label": 0,
        "header": False,
        "task": "classification",
        "id": "SOY",
    },
    "soybean-large": {
        "label": 0,
        "header": False,
        "task": "classification",
        "id": "SOY",
    },
    # 6. Car Evaluation — ocena jakości samochodu
    # Plik: car.data (brak nagłówków), etykieta: class (unacc/acc/good/v-good)
    # Cechy: buying, maint, doors, persons, lug_boot, safety
    "car+evaluation": {
        "label": "class",
        "header": False,
        "task": "classification",
        "id": "CAR",
        "columns": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
    },
    # 7. Statlog (Vehicle Silhouettes) — klasyfikacja sylwetek pojazdów
    # Pliki: xaa.dat – xai.dat (9 plików, skrypt scali automatycznie)
    # Etykieta: ostatnia kolumna (indeks 18) — 4 klasy: bus, saab, opel, van
    # 846 próbek łącznie, 18 cech numerycznych, separator spacja, brak nagłówków
    "statlog+vehicle+silhouettes": {
        "label": 18,
        "header": False,
        "task": "classification",
        "sep": " ",
        "id": "SVS",
    },
    # 8. Dry Bean — klasyfikacja odmian fasoli (obraz + cechy kształtu)
    # Plik: Dry_Bean_Dataset.xlsx — etykieta: Class (7 odmian)
    # 13611 próbek, 16 cech numerycznych, brak brakujących wartości
    "dry+bean+dataset": {
        "label": "Class",
        "header": True,
        "task": "classification",
        "id": "DRY",
    },
    "dry_bean_dataset": {
        "label": "Class",
        "header": True,
        "task": "classification",
        "id": "DRY",
    },
    # 9. Crowdsourced Mapping — klasyfikacja użytkowania terenu (NDVI satelitarny)
    # Pliki: training.csv + testing.csv — skrypt scali je automatycznie (glob w folderze)
    # Etykieta: class (forest/farm/impervious/grass/water/orchard) — 6 klas
    # Łącznie: ~10845 próbek, 28 cech (wartości NDVI z różnych dat)
    "crowdsourced+mapping": {
        "label": "class",
        "header": True,
        "task": "classification",
        "id": "CM",
    },
    # 10. Avian influenza A(H5N1) — dane epidemiologiczne (plik .xls)
    # Kolumny: id, year, country, conf_status, age, sex, hospital, outcome, exposure, ...
    # Etykieta: outcome (fatal / cured / under treatment / n.d.)
    # Uwaga: header=1 bo wiersz 0 to tytuł artykułu, wiersz 1 to nagłówki kolumn
    "avian+influenza+h5n1": {
        "label": "outcome",
        "header": 1,
        "task": "classification",
        "id": "AIH",
        "drop_columns": ["id"],
    },
    "aviaere_influenza_fallliste": {
        "label": "outcome",
        "header": 1,
        "task": "classification",
        "id": "AIH",
        "drop_columns": ["id"],
    },
}

# Mapowanie akronimu na nazwę datasetu (do logowania)
DATASET_NAMES = {
    "RT":   "rt-iot2022",
    "LR":   "letter+recognition",
    "UNSW": "unsw-nb15",
    "SLS":  "statlog+landsat+satellite",
    "SOY":  "soybean+large",
    "CAR":  "car+evaluation",
    "SVS":  "statlog+vehicle+silhouettes",
    "DRY":  "dry+bean+dataset",
    "CM":   "crowdsourced+mapping",
    "AIH":  "aviaere_influenza_fallliste",
}

# Znane nazwy kolumn etykiet (rozszerzona lista do automatycznego wykrywania)
KNOWN_LABEL_NAMES = [
    'target', 'label', 'class', 'y', 'output', 'outcome',
    'diagnosis', 'attack_type', 'category', 'type',
    'species', 'variety', 'action', 'result',
    'klasa', 'etykieta', 'cel',
    'diabetes_012', 'diabetes_binary', 'diabetes',
    'disposition', 'status', 'Class', 'lettr',
]
# ============================================


def get_dataset_config(name: str) -> dict:
    """
    Zwraca konfigurację datasetu na podstawie nazwy folderu/pliku.
    Sprawdza dokładne dopasowanie i częściowe (zawieranie).
    """
    # Dokładne dopasowanie
    if name in DATASET_CONFIG:
        return DATASET_CONFIG[name]
    
    # Dopasowanie bez rozszerzenia
    name_no_ext = Path(name).stem
    if name_no_ext in DATASET_CONFIG:
        return DATASET_CONFIG[name_no_ext]
    
    # Częściowe dopasowanie (nazwa folderu zawiera klucz lub odwrotnie)
    name_lower = name.lower()
    for key, config in DATASET_CONFIG.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return config
    
    return None


def _detect_no_header(df: pd.DataFrame) -> bool:
    """
    Sprawdza czy DataFrame został wczytany bez nagłówków
    (tzn. pierwszy wiersz danych trafił jako nazwy kolumn).
    Heurystyka: jeśli większość nazw kolumn wygląda jak wartości liczbowe.
    """
    numeric_col_count = 0
    for col in df.columns:
        col_str = str(col).strip()
        try:
            float(col_str)
            numeric_col_count += 1
        except ValueError:
            pass
    
    # Jeśli >50% kolumn to liczby, prawdopodobnie brak nagłówka
    return numeric_col_count > len(df.columns) * 0.5


def _read_file(file_path: Path, config: dict = None) -> pd.DataFrame:
    """
    Wczytuje pojedynczy plik z uwzględnieniem konfiguracji datasetu.
    """
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
            # header=1 dla AIH (wiersz 0 to tytul), header=0 to domyslne
            header = cfg_header
        elif cfg_header is True:
            header = 0  # jawnie: pierwszy wiersz to naglowki
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.csv', '.data']:
        df = pd.read_csv(file_path, sep=sep, header=header, names=names)
    elif suffix in ['.trn', '.tst']:
        # Pliki treningowe/testowe Statlog — separator spacja
        df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=header, names=names)
    elif suffix == '.txt':
        if sep != ',':
            df = pd.read_csv(file_path, sep=sep, header=header, names=names)
        else:
            df = pd.read_csv(file_path, sep=None, engine='python', header=header, names=names)
    elif suffix == '.dat':
        # Pliki .dat często używają separatora whitespace (spacja/tab/wielokrotna spacja)
        df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=header, names=names)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, header=header, names=names)
    elif suffix == '':
        df = pd.read_csv(file_path, sep=sep, header=header, names=names)
    else:
        df = pd.read_csv(file_path, sep=sep, header=header, names=names)
    
    # Automatyczne wykrywanie brakujących nagłówków (jeśli nie było konfiguracji)
    if config is None and _detect_no_header(df):
        logger.warning(f"  Wykryto brak nagłówków w {file_path.name} - wczytuję ponownie")
        if suffix == '.txt':
            df = pd.read_csv(file_path, sep=None, engine='python', header=None)
        else:
            df = pd.read_csv(file_path, sep=sep, header=None)
    
    return df


def _clean_name(name: str) -> str:
    """
    Usuwa rozszerzenia plików z nazwy datasetu.
    Np. 'diabetes_012_health_indicators_BRFSS2015.csv' -> 'diabetes_012_health_indicators_BRFSS2015'
    Zapobiega podwójnym rozszerzeniom w nazwach plików wyjściowych.
    """
    known_extensions = ['.csv', '.data', '.txt', '.xlsx', '.xls', '.zip']
    while any(name.lower().endswith(ext) for ext in known_extensions):
        for ext in known_extensions:
            if name.lower().endswith(ext):
                name = name[:-len(ext)]
                break
    return name


def load_dataset(data_path: str) -> tuple:
    """
    Wczytuje zbiór danych z różnych formatów (CSV, TXT, Excel, ZIP, folder).
    Uwzględnia konfigurację z DATASET_CONFIG.
    """
    path = Path(data_path)
    
    # Pobierz konfigurację datasetu
    config = get_dataset_config(path.name)
    if config:
        logger.info(f"Znaleziono konfigurację dla '{path.name}'")
    
    if path.is_file():
        # Obsługa plików ZIP
        if path.suffix.lower() == '.zip':
            return load_from_zip(path, config)
        else:
            df = _read_file(path, config)
            logger.info(f"Wczytano plik: {path.name} ({len(df)} wierszy)")
        name = _clean_name(path.stem if path.stem else path.name)
    elif path.is_dir():
        # Lista plików do ignorowania
        IGNORE_FILES_LOWER = ['.gitignore', '.ds_store', 'index', 'license', 'readme',
                              'sha256sums', 'checksums', 'manifest', 'license.txt',
                              'readme.txt', 'sha256sums.txt', 'checksums.txt',
                              # FIX: description files of soybean+large / statlog must not be read as data
                              'vehicle.doc', 'sat.doc', 'soybean-explanation',
                              'stepp-order', 'fisher-order', 'why-various-soybean-databases']
        
        # Szukaj rekursywnie w folderze i podfolderach
        data_files = (list(path.glob('**/*.csv')) + list(path.glob('**/*.data')) + 
                      list(path.glob('**/*.dat')) + list(path.glob('**/*.trn')) + 
                      list(path.glob('**/*.tst')) + list(path.glob('**/*.zip')) +
                      list(path.glob('**/*.xls')) + list(path.glob('**/*.xlsx')))
        
        # Dodaj pliki .txt ale filtruj metadane
        txt_files = list(path.glob('**/*.txt'))
        data_files += [f for f in txt_files if f.name.lower() not in IGNORE_FILES_LOWER]
        
        # Dodaj pliki bez rozszerzenia (które mogą być CSV)
        for f in path.rglob('*'):
            if f.is_file() and f.suffix == '' and f.name.lower() not in IGNORE_FILES_LOWER:
                data_files.append(f)
        
        # Filtruj pliki metadanych
        data_files = [f for f in data_files if f.name.lower() not in IGNORE_FILES_LOWER]
        
        if not data_files:
            raise ValueError(f"Nie znaleziono plików danych w folderze: {path}")
        
        dfs = []
        for data_file in data_files:
            if data_file.name.lower() in IGNORE_FILES_LOWER:
                continue
            try:
                if data_file.suffix.lower() == '.zip':
                    df_temp, _ = load_from_zip(data_file, config)
                    dfs.append(df_temp)
                    logger.info(f"Wczytano z ZIP: {data_file.name} ({len(df_temp)} wierszy)")
                else:
                    df_temp = _read_file(data_file, config)
                    dfs.append(df_temp)
                    logger.info(f"Wczytano plik: {data_file.name} ({len(df_temp)} wierszy)")
            except Exception as e:
                logger.warning(f"Nie udało się wczytać {data_file}: {e}")
        
        if not dfs:
            raise ValueError(f"Nie udało się wczytać żadnego pliku z {path}")
        
        df = pd.concat(dfs, ignore_index=True)
        name = _clean_name(path.name)
    else:
        raise ValueError(f"Ścieżka nie istnieje: {path}")
    
    # Usuwanie kolumn wskazanych w konfiguracji (np. ID)
    if config and 'drop_columns' in config:
        cols_to_drop = [c for c in config['drop_columns'] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Usunięto kolumny: {cols_to_drop}")
    
    # Usuwanie wierszy z brakującymi wartościami (NaN) i '?'
    df = df.replace('?', np.nan)
    
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    if rows_before != rows_after:
        logger.info(f"Usunięto {rows_before - rows_after} wierszy z brakującymi wartościami")
    
    # Ujednolicenie typow w kolumnie etykiet (naprawia blad SOY: int vs str)
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
    
    logger.info(f"Wczytano zbior '{name}': {len(df)} probek, {len(df.columns)} cech")
    
    return df, name


def load_from_zip(zip_path: Path, config: dict = None) -> tuple:
    """
    Wczytuje dane z pliku ZIP.
    """
    dfs = []
    name = _clean_name(zip_path.stem)
    
    sep = ','
    header = 'infer'
    names = None
    if config:
        sep = config.get('sep', ',')
        if config.get('header') is False:
            header = None
            names = config.get('columns', None)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_name in zf.namelist():
            # Pomijamy foldery
            if file_name.endswith('/'):
                continue
            
            # Pomijamy pliki systemowe i metadane
            base_name = Path(file_name).name
            if base_name.startswith('.') or base_name.lower() in [
                'license', 'readme', 'license.txt', 'readme.txt', 
                'index', 'sha256sums.txt', 'checksums.txt', 'manifest.txt'
            ]:
                continue
            
            try:
                with zf.open(file_name) as f:
                    if file_name.lower().endswith('.csv') or file_name.lower().endswith('.data'):
                        df_temp = pd.read_csv(f, sep=sep, header=header, names=names)
                    elif file_name.lower().endswith('.txt'):
                        if sep != ',':
                            df_temp = pd.read_csv(f, sep=sep, header=header, names=names)
                        else:
                            df_temp = pd.read_csv(f, sep=None, engine='python', header=header, names=names)
                    elif file_name.lower().endswith(('.xlsx', '.xls')):
                        df_temp = pd.read_excel(f, header=header, names=names)
                    else:
                        df_temp = pd.read_csv(f, sep=sep, header=header, names=names)
                    
                    dfs.append(df_temp)
                    logger.info(f"Wczytano z ZIP: {file_name} ({len(df_temp)} wierszy)")
            except Exception as e:
                logger.warning(f"Nie udało się wczytać {file_name} z ZIP: {e}")
    
    if not dfs:
        raise ValueError(f"Nie znaleziono danych w pliku ZIP: {zip_path}")
    
    df = pd.concat(dfs, ignore_index=True)
    return df, name


def detect_label_column(df: pd.DataFrame, dataset_name: str = None) -> str:
    """
    Automatycznie wykrywa kolumnę z etykietami klas.
    
    Priorytet:
    1. Konfiguracja z DATASET_CONFIG (po nazwie datasetu)
    2. Dopasowanie nazwy kolumny do znanych etykiet
    3. Kolumna kategoryczna (object/string) z najmniejszą liczbą unikalnych wartości
    4. Ostatnia kolumna z małą liczbą unikalnych wartości
    5. Ostatnia kolumna (fallback)
    """
    # 1. Sprawdź konfigurację datasetu
    if dataset_name:
        config = get_dataset_config(dataset_name)
        if config and 'label' in config:
            label = config['label']
            if isinstance(label, str) and label in df.columns:
                logger.info(f"  Etykieta z konfiguracji: '{label}'")
                return label
            elif isinstance(label, int) and label < len(df.columns):
                col_name = df.columns[label]
                logger.info(f"  Etykieta z konfiguracji (indeks {label}): '{col_name}'")
                return col_name
    
    # 2. Szukaj po znanych nazwach kolumn
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for known_name in KNOWN_LABEL_NAMES:
        if known_name.lower() in df_cols_lower:
            matched_col = df_cols_lower[known_name.lower()]
            logger.info(f"  Etykieta wykryta po nazwie: '{matched_col}'")
            return matched_col
    
    # 2b. Częściowe dopasowanie (kolumna zawiera znane słowo)
    label_keywords = ['target', 'label', 'class', 'diagnosis', 'attack', 'category', 'outcome']
    for col in df.columns:
        col_lower = col.lower().strip()
        for keyword in label_keywords:
            if keyword in col_lower:
                logger.info(f"  Etykieta wykryta po słowie kluczowym '{keyword}': '{col}'")
                return col
    
    # 3. Szukaj kolumn kategorycznych (object/string) - najprawdopodobniej etykiety
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) == 1:
        logger.info(f"  Etykieta wykryta: jedyna kolumna kategoryczna '{categorical_cols[0]}'")
        return categorical_cols[0]
    elif len(categorical_cols) > 1:
        # Wybierz kategoryczną z najmniejszą liczbą unikalnych wartości
        min_unique = float('inf')
        best_col = categorical_cols[-1]
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if n_unique < min_unique:
                min_unique = n_unique
                best_col = col
        logger.info(f"  Etykieta wykryta: kolumna kategoryczna z min. unikalnymi '{best_col}' ({min_unique} klas)")
        return best_col
    
    # 4. Ostatnia kolumna z małą liczbą unikalnych wartości
    last_col = df.columns[-1]
    first_col = df.columns[0]
    
    # Sprawdź ostatnią kolumnę
    if df[last_col].nunique() <= 50:
        logger.info(f"  Etykieta: ostatnia kolumna '{last_col}' ({df[last_col].nunique()} unikalnych)")
        return last_col
    
    # Sprawdź pierwszą kolumnę (niektóre datasety mają etykietę na początku)
    if df[first_col].nunique() <= 50:
        logger.info(f"  Etykieta: pierwsza kolumna '{first_col}' ({df[first_col].nunique()} unikalnych)")
        return first_col
    
    # 5. Fallback - ostatnia kolumna
    logger.warning(f"  Nie udało się jednoznacznie wykryć etykiety, używam ostatniej kolumny: '{last_col}'")
    return last_col


def is_classification(df: pd.DataFrame, label_column: str) -> bool:
    """
    Sprawdza czy problem jest klasyfikacyjny (dyskretne klasy).
    """
    y = df[label_column]
    
    # Jeśli kolumna jest typu object/string - klasyfikacja
    if y.dtype == 'object' or y.dtype.name == 'category':
        return True
    
    # Jeśli liczba unikalnych wartości jest mała względem rozmiaru - klasyfikacja
    n_unique = y.nunique()
    if n_unique <= 50 and n_unique < len(df) * 0.1:
        return True
    
    # Jeśli wartości są całkowite i jest ich mało - klasyfikacja
    if np.allclose(y.dropna(), y.dropna().astype(int)) and n_unique <= 50:
        return True
    
    return False


def stratified_kfold_split(df: pd.DataFrame, label_column: str, n_splits: int, 
                           random_state: int = 42) -> list:
    """
    KFold ze stratyfikacją - zachowuje proporcje klas.
    Fallback do losowego podziału gdy stratyfikacja niemożliwa.
    """
    try:
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        subsets = []
        for _, test_idx in skf.split(X, y):
            subset_df = df.iloc[test_idx].copy()
            subsets.append(subset_df)
        
        return subsets
    except ValueError as e:
        logger.warning(f"Stratyfikacja KFold niemożliwa: {e}")
        logger.warning(f"Używam losowego KFold (bez stratyfikacji)")
        return random_kfold_split(df, n_splits, random_state)


def random_kfold_split(df: pd.DataFrame, n_splits: int, random_state: int = 42) -> list:
    """
    5CV bez stratyfikacji - losowy podział.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    subsets = []
    for _, test_idx in kf.split(df):
        subset_df = df.iloc[test_idx].copy()
        subsets.append(subset_df)
    
    return subsets


def stratified_split(df: pd.DataFrame, label_column: str, n_splits: int, 
                     random_state: int = 42) -> list:
    """
    Dzieli zbiór na n rozłącznych podzbiorów ze stratyfikacją.
    Fallback do losowego podziału gdy stratyfikacja niemożliwa.
    """
    if n_splits == 1:
        return [df.copy()]
    
    try:
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        subsets = []
        for _, test_idx in skf.split(X, y):
            subset_df = df.iloc[test_idx].copy()
            subsets.append(subset_df)
        
        return subsets
    except ValueError as e:
        logger.warning(f"Stratyfikacja split({n_splits}) niemożliwa: {e}")
        logger.warning(f"Używam losowego podziału (bez stratyfikacji)")
        return random_split(df, n_splits, random_state)


def random_split(df: pd.DataFrame, n_splits: int, random_state: int = 42) -> list:
    """
    Dzieli zbiór na n rozłącznych podzbiorów losowo.
    """
    if n_splits == 1:
        return [df.copy()]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    subsets = []
    for _, test_idx in kf.split(df):
        subset_df = df.iloc[test_idx].copy()
        subsets.append(subset_df)
    
    return subsets


def stratified_val_test_split(df: pd.DataFrame, label_column: str, 
                               random_state: int = 42) -> tuple:
    """
    Dzieli zbiór na val (50%) i test (50%) ze stratyfikacją.
    Zachowuje oryginalną kolejność kolumn.
    Jeśli stratyfikacja nie jest możliwa (za mało próbek w klasie), używa losowego podziału.
    """
    y = df[label_column]
    
    try:
        idx_val, idx_test = train_test_split(
            df.index, test_size=0.5, stratify=y, random_state=random_state
        )
        df_val = df.loc[idx_val].copy()
        df_test = df.loc[idx_test].copy()
    except ValueError as e:
        # Fallback do losowego podziału gdy stratyfikacja niemożliwa
        logger.warning(f"Stratyfikacja niemożliwa, używam losowego podziału: {e}")
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        mid = len(df_shuffled) // 2
        df_val = df_shuffled.iloc[:mid].copy()
        df_test = df_shuffled.iloc[mid:].copy()
    
    return df_val, df_test


def random_val_test_split(df: pd.DataFrame, random_state: int = 42) -> tuple:
    """
    Dzieli zbiór na val (50%) i test (50%) losowo.
    """
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    mid = len(df_shuffled) // 2
    
    df_val = df_shuffled.iloc[:mid].copy()
    df_test = df_shuffled.iloc[mid:].copy()
    
    return df_val, df_test


def create_federated_datasets(data_path: str, output_base_path: str, 
                               acronym: str = "name",
                               label_column: str = None,
                               use_stratification: bool = None,
                               random_state: int = None):
    """
    Realizacja pseudokodu:
    
    1. Utwórz folder fname
    2. 5CV: D = D1 u D2 u … u D5
    3. For i:=1 to 5:
       - Di -> Dival (50%) + Dits (50%) [stratyfikacja]
       - Ditr = D \\ Di
       - Utwórz folder fname/f{acronym}_fold_i
    4. For i:=1 to 5:
       For j:=3 to 10:
       - Podziel Ditr na j podzbiorów [stratyfikacja]
       - Utwórz folder fname/f{acronym}_fold_i/j_federation
       - Zapisz pliki val, test, tr_1..tr_j
    
    Parametr acronym definiuje skrót nazwy datasetu używany w nazwach
    folderów i plików (np. "RT", "LR", "UNSW", "SLS", "SOY", "CAR", "SVS",
    "DRY", "CM", "AIH", "MED").
    """
    # Użyj globalnego RANDOM_SEED jeśli nie podano
    if random_state is None:
        random_state = RANDOM_SEED
    # Wczytaj dane
    df, name = load_dataset(data_path)
    
    # fname = akronim datasetu (np. "BCW", "DHI" itd.)
    fname = acronym
    
    # Wykryj kolumnę etykiet (z uwzględnieniem konfiguracji datasetu)
    if label_column is None:
        label_column = detect_label_column(df, dataset_name=name)
    
    if label_column not in df.columns:
        raise ValueError(f"Kolumna '{label_column}' nie istnieje. "
                        f"Dostępne kolumny: {list(df.columns)}")

    # Wymuś jednolity typ etykiety (str) — naprawia SOY (mieszane int/str -> błąd sortowania w np.unique)
    df[label_column] = df[label_column].astype(str)

    # Sprawdź czy użyć stratyfikacji
    # Priorytet: parametr -> konfiguracja datasetu -> automatyczne wykrywanie
    if use_stratification is None:
        config = get_dataset_config(name)
        if config and 'task' in config:
            use_stratification = (config['task'] == 'classification')
            logger.info(f"Typ zadania z konfiguracji: {config['task']}")
        else:
            use_stratification = is_classification(df, label_column)
    
    if use_stratification:
        logger.info(f"Wykryto problem KLASYFIKACJI - używam stratyfikacji")
        logger.info(f"Kolumna etykiet: '{label_column}'")
    else:
        logger.info(f"Wykryto problem REGRESJI - używam losowego podziału")
    
    # === Folder wyjściowy (np. data_splits/Breast_Cancer_Wisconsin(BCW)) ===
    output_path = Path(output_base_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Folder wyjściowy: {output_path}")
    
    # === KCV: D = D1 u D2 u … u Dk ===
    n_folds = N_FOLDS
    if use_stratification:
        folds = stratified_kfold_split(df, label_column, n_folds, random_state)
    else:
        folds = random_kfold_split(df, n_folds, random_state)
    
    logger.info(f"\n{n_folds}CV podział:")
    for i, fold in enumerate(folds, 1):
        logger.info(f"  D{i}: {len(fold)} próbek")
    
    # === Przygotuj struktury danych dla każdego folda ===
    fold_data = {}
    
    # For i:=1 to 5 do
    for i in range(1, n_folds + 1):
        logger.info(f"\nPrzygotowanie Fold {i}")
        
        # Di
        Di = folds[i-1]
        
        # Ditr = D \ Di
        Ditr = pd.concat([folds[j] for j in range(n_folds) if j != i-1], ignore_index=True)
        
        # Podział Di na Dival (50%) oraz Dits (50%)
        if use_stratification:
            try:
                Di_val, Di_test = stratified_val_test_split(Di, label_column, random_state + i)
            except ValueError as e:
                logger.warning(f"  Stratyfikacja nie możliwa dla val/test split: {e}")
                logger.warning(f"  Używam losowego podziału dla Fold {i}")
                Di_val, Di_test = random_val_test_split(Di, random_state + i)
        else:
            Di_val, Di_test = random_val_test_split(Di, random_state + i)
        
        logger.info(f"  |Di|={len(Di)}, |Dival|={len(Di_val)} (50%), |Dits|={len(Di_test)} (50%)")
        logger.info(f"  |Ditr|={len(Ditr)}")
        
        # Utwórz folder f{acronym}_fold_i (np. fBCW_fold_1)
        fold_path = output_path / f"f{fname}_fold_{i}"
        fold_path.mkdir(parents=True, exist_ok=True)
        
        # Zapisz dane folda
        fold_data[i] = {
            'Di_val': Di_val,
            'Di_test': Di_test,
            'Ditr': Ditr,
            'fold_path': fold_path
        }
    
    # === Tworzenie federacji ===
    # For i:=1 to 5 do
    for i in range(1, n_folds + 1):
        logger.info(f"\nTworzenie federacji dla Fold {i}")
        
        Di_val = fold_data[i]['Di_val']
        Di_test = fold_data[i]['Di_test']
        Ditr = fold_data[i]['Ditr']
        fold_path = fold_data[i]['fold_path']
        
        # For j:=2 to 5 do
        for j in range(MIN_CLIENTS, MAX_CLIENTS + 1):
            logger.info(f"  Federacja j={j} klientów")
            
            # Podział Ditr na j rozłącznych podzbiorów
            if use_stratification:
                Ditr_subsets = stratified_split(Ditr, label_column, j, random_state + i * 10 + j)
            else:
                Ditr_subsets = random_split(Ditr, j, random_state + i * 10 + j)
            
            # Utwórz folder f{acronym}_fold_i/j_federation
            federation_path = fold_path / f"{j}_federation"
            federation_path.mkdir(parents=True, exist_ok=True)
            
            # f{acronym}_fold_i_j_federation_val.csv
            val_filename = f"f{fname}_fold_{i}_{j}_federation_val.csv"
            Di_val.to_csv(federation_path / val_filename, index=False)
            
            # f{acronym}_fold_i_j_federation_test.csv
            test_filename = f"f{fname}_fold_{i}_{j}_federation_test.csv"
            Di_test.to_csv(federation_path / test_filename, index=False)
            
            # For t:=1 to j do - pliki treningowe
            for t in range(1, j + 1):
                train_filename = f"f{fname}_fold_{i}_{j}_federation_tr_{t}.csv"
                Ditr_subsets[t-1].to_csv(federation_path / train_filename, index=False)
                logger.info(f"    tr_{t}: {len(Ditr_subsets[t-1])} próbek")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"ZAKOŃCZONO: {fname}")
    logger.info(f"Wyniki: {output_path}")
    logger.info(f"{'='*50}")
    
    return output_path


def main():
    base = Path(BASE_PATH)
    
    for idx, (input_rel, output_rel, acronym) in enumerate(DATASETS, 1):
        input_path = base / input_rel
        output_path = base / output_rel
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"DATASET {idx}/{len(DATASETS)}: {input_path.name} (akronim: {acronym})")
        logger.info(f"  Wejście: {input_path}")
        logger.info(f"  Wyjście: {output_path}")
        logger.info(f"{'#'*60}")
        
        try:
            create_federated_datasets(str(input_path), str(output_path), acronym=acronym)
        except Exception as e:
            logger.error(f"Błąd: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ZAKOŃCZONO WSZYSTKIE DATASETY ({len(DATASETS)})")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
