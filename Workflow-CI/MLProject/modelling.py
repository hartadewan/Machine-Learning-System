from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.environment_variables import MLFLOW_RUN_ID
from sklearn.ensemble import RandomForestClassifier

# ===========
# Konfigurasi
# ===========
LOGGER = logging.getLogger(__name__)

TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
EXPERIMENT_NAME: str = "Credit Scoring - Basic (Autolog)"
DATA_DIR_NAME: str = "namadataset_preprocessing"
TARGET_COLUMN: str = "credit_risk"
RANDOM_STATE: int = 42


def resolve_data_dir(dir_name: str = DATA_DIR_NAME) -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (Path(dir_name), here / dir_name, here.parent / dir_name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Folder '{dir_name}' tidak ditemukan.")


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data_dir = resolve_data_dir()
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    LOGGER.info("Data dimuat — latih: %s, uji: %s.", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def configure_tracking() -> None:

    mlflow.set_tracking_uri(TRACKING_URI)
    if MLFLOW_RUN_ID.defined:
        LOGGER.info("Dijalankan via MLflow Project (run %s) | Tracking URI: %s",
                    MLFLOW_RUN_ID.get(), TRACKING_URI)
        return
    mlflow.set_experiment(EXPERIMENT_NAME)
    LOGGER.info("MLflow Tracking URI: %s | Eksperimen: %s",
                TRACKING_URI, EXPERIMENT_NAME)


def train() -> None:
    configure_tracking()

    # Autolog
    mlflow.sklearn.autolog()

    X_train, y_train, X_test, y_test = load_dataset()

    with mlflow.start_run(run_name="random_forest_basic"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        # Evaluasi ringkas
        test_accuracy = model.score(X_test, y_test)
        LOGGER.info("Akurasi set uji: %.4f", test_accuracy)

    LOGGER.info("Pelatihan Basic selesai. Buka %s untuk melihat dashboard.",
                TRACKING_URI)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()


if __name__ == "__main__":
    main()
