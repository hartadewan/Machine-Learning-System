from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils import estimator_html_repr

# ───────────
# Konfigurasi
# ───────────
LOGGER = logging.getLogger(__name__)

# --- Pelacakan: DagsHub (Advance) ---
USE_DAGSHUB: bool = os.environ.get("USE_DAGSHUB", "true").lower() in {"1", "true", "yes"}
DAGSHUB_REPO_OWNER: str = os.environ.get("DAGSHUB_REPO_OWNER", "<hartadewan>")
DAGSHUB_REPO_NAME: str = os.environ.get("DAGSHUB_REPO_NAME", "<mlflow-tracking-ui>")
LOCAL_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")

EXPERIMENT_NAME: str = "Credit Scoring - Tuning (Manual)"
DATA_DIR_NAME: str = "namadataset_preprocessing"
TARGET_COLUMN: str = "credit_risk"
RANDOM_STATE: int = 42

PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}


# ────────────────────
# Load data & tracking
# ────────────────────
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

    if USE_DAGSHUB:
        import dagshub
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER,
                     repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        LOGGER.info("Pelacakan online DagsHub: %s/%s",
                    DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME)
    else:
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
        LOGGER.info("Pelacakan lokal: %s", LOCAL_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ─────────────────
# Pembuatan artefak
# ─────────────────
def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


def log_skilled_artifacts(model, metrics, y_test, y_pred, art_dir: Path) -> None:
 
    (art_dir / "estimator.html").write_text(estimator_html_repr(model),
                                            encoding="utf-8")
    mlflow.log_artifact(str(art_dir / "estimator.html"))


    mlflow.log_dict(metrics, "metric_info.json")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["good (0)", "bad (1)"], ax=ax,
        colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = art_dir / "training_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))


def log_advance_artifacts(model, X_test, y_test, y_pred, y_proba,
                          art_dir: Path) -> None:

    # 1) Kurva ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve"); fig.tight_layout()
    roc_path = art_dir / "roc_curve.png"; fig.savefig(roc_path, dpi=120)
    plt.close(fig); mlflow.log_artifact(str(roc_path))

    # 2) Kurva Precision-Recall
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Precision-Recall Curve"); fig.tight_layout()
    pr_path = art_dir / "precision_recall_curve.png"; fig.savefig(pr_path, dpi=120)
    plt.close(fig); mlflow.log_artifact(str(pr_path))

    # 3) Feature importance (15 teratas)
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top = importances.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Feature Importance"); fig.tight_layout()
    fi_path = art_dir / "feature_importance.png"; fig.savefig(fi_path, dpi=120)
    plt.close(fig); mlflow.log_artifact(str(fi_path))

    # 4) Classification report (teks)
    report = classification_report(y_test, y_pred,
                                   target_names=["good (0)", "bad (1)"])
    (art_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    mlflow.log_artifact(str(art_dir / "classification_report.txt"))


# ──────────────────────
# Training Orchestration
# ────────────────────── 
def train() -> None:

    configure_tracking()
    X_train, y_train, X_test, y_test = load_dataset()

    with mlflow.start_run(run_name="random_forest_tuned"):
        # --- Hyperparameter tuning ---
        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid=PARAM_GRID, cv=5, scoring="f1", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        LOGGER.info("Parameter terbaik: %s", grid.best_params_)

        # --- Prediksi & metrik pada set uji ---
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # --- Manual logging: parameter, metrik, model ---
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metric("best_cv_f1", grid.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

        # --- Artefak Skilled + Advance ---
        with tempfile.TemporaryDirectory() as tmp:
            art_dir = Path(tmp)
            log_skilled_artifacts(best_model, metrics, y_test, y_pred, art_dir)
            log_advance_artifacts(best_model, X_test, y_test, y_pred,
                                  y_proba, art_dir)

        LOGGER.info("Metrik uji: %s",
                    {k: round(v, 4) for k, v in metrics.items()})

    LOGGER.info("Pelatihan Skilled/Advance selesai.")


def main() -> None:
    """Titik masuk eksekusi skrip."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()


if __name__ == "__main__":
    main()

