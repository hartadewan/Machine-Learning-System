from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# Configuration & Constanta
# =========================
LOGGER = logging.getLogger(__name__)

COLUMN_NAMES: list[str] = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property_type",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target",
]

NUMERIC_FEATURES: list[str] = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "age_years", "existing_credits", "num_dependents",
]

RAW_TARGET_COLUMN: str = "target"        
TARGET_COLUMN: str = "credit_risk"       
TARGET_MAPPING: dict[int, int] = {1: 0, 2: 1}   

DEFAULT_RAW_FILENAME: str = "credit_scoring_raw.csv"
DEFAULT_OUTPUT_DIR: Path = Path("credit_scoring_preprocessing")
DEFAULT_TEST_SIZE: float = 0.20
DEFAULT_RANDOM_STATE: int = 42


# ===================
# Preprocessing Steps
# ===================
def resolve_raw_path(filename: str = DEFAULT_RAW_FILENAME) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        Path("..") / filename,
        Path("namadataset_raw") / filename,
        Path("..") / "namadataset_raw" / filename,
        here / filename,
        here.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"'{filename}' tidak ditemukan. Lokasi yang dicek: "
        + ", ".join(str(c) for c in candidates)
    )


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    raw_path = Path(path) if path is not None else resolve_raw_path()
    df = pd.read_csv(raw_path, header=None, names=COLUMN_NAMES)
    LOGGER.info("Data mentah dimuat dari %s (%d baris, %d kolom).",
                raw_path, df.shape[0], df.shape[1])
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result[TARGET_COLUMN] = result[RAW_TARGET_COLUMN].map(TARGET_MAPPING)
    result = result.drop(columns=[RAW_TARGET_COLUMN])
    LOGGER.info("Target dipetakan ke '%s' -> %s.", TARGET_COLUMN,
                result[TARGET_COLUMN].value_counts().sort_index().to_dict())
    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    LOGGER.info("Baris duplikat dihapus: %d.", before - len(cleaned))

    n_missing = int(cleaned.isna().sum().sum())
    if n_missing:
        raise ValueError(f"Data masih memiliki {n_missing} nilai kosong.")
    LOGGER.info("Pemeriksaan nilai kosong lolos (0 nilai kosong).")
    return cleaned


def identify_feature_types(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in NUMERIC_FEATURES if c in features.columns]
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    LOGGER.info("Fitur numerik: %d | fitur kategorikal: %d.",
                len(numeric_cols), len(categorical_cols))
    return numeric_cols, categorical_cols


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    LOGGER.info("Pemisahan data — latih: %d baris, uji: %d baris "
                "(proporsi bad latih=%.3f, uji=%.3f).",
                len(X_train), len(X_test), y_train.mean(), y_test.mean())
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_arr = preprocessor.fit_transform(X_train)   # fit HANYA di set latih
    X_test_arr = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_prep = pd.DataFrame(X_train_arr, columns=feature_names,
                                index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_arr, columns=feature_names,
                               index=X_test.index)
    LOGGER.info("Transformasi selesai — %d fitur setelah encoding.",
                X_train_prep.shape[1])
    return X_train_prep, X_test_prep


def assemble_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    dataset = features.copy()
    dataset[TARGET_COLUMN] = target.values
    return dataset


def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    LOGGER.info("Artefak tersimpan di %s (train=%s, test=%s).",
                out_dir.resolve(), train_df.shape, test_df.shape)
    return out_dir


# =============
# Orchestration
# =============
def preprocess_data(
    raw_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Memulai pipeline preprocessing...")

    df = load_raw_data(raw_path)                       
    df = encode_target(df)                             
    df = clean_data(df)                                

    features = df.drop(columns=[TARGET_COLUMN])        
    target = df[TARGET_COLUMN]
    numeric_cols, categorical_cols = identify_feature_types(features)

    X_train, X_test, y_train, y_test = split_data(     
        features, target, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)   
    X_train_prep, X_test_prep = transform_features(preprocessor, X_train, X_test)

    train_df = assemble_dataset(X_train_prep, y_train)  
    test_df = assemble_dataset(X_test_prep, y_test)

    if save:
        save_datasets(train_df, test_df, output_dir)

    LOGGER.info("Pipeline selesai. Data siap latih dikembalikan.")
    return train_df, test_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset credit scoring (MSML)."
    )
    parser.add_argument("--input", default=None,
                        help="Path berkas data mentah (default: resolve otomatis).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                        help="Folder output artefak preprocessing.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help="Proporsi data uji (default: 0.20).")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE,
                        help="Seed reproduktibilitas (default: 42).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    preprocess_data(
        raw_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
