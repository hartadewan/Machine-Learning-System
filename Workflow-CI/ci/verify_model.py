from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

LOGGER = logging.getLogger("verify_model")
ALLOWED_LABELS = {0, 1}


class ModelVerificationError(RuntimeError):

    pass


def load_holdout(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:

    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ModelVerificationError(f"Target column '{target}' not found in {path}.")
    if frame.empty:
        raise ModelVerificationError(f"{path} contains no rows.")
    return frame.drop(columns=[target]), frame[target]


def score(model_uri: str, features: pd.DataFrame, labels: pd.Series) -> dict:

    model = mlflow.pyfunc.load_model(model_uri)
    predictions = np.asarray(model.predict(features)).ravel()
    if predictions.shape[0] != len(features):
        raise ModelVerificationError(
            f"Got {predictions.shape[0]} predictions for {len(features)} rows.")
    observed = {int(value) for value in np.unique(predictions)}
    if not observed <= ALLOWED_LABELS:
        raise ModelVerificationError(f"Predictions contain non-binary labels: {observed}.")
    return {
        "model_uri": model_uri,
        "rows": int(len(features)),
        "holdout_accuracy": round(float((predictions == labels.to_numpy()).mean()), 6),
        "predicted_positive_rate": round(float(predictions.mean()), 6),
        "labels_seen": sorted(observed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--data", type=Path, required=True, help="Preprocessed test.csv.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    try:
        features, labels = load_holdout(args.data, args.target)
        report = score(args.model_uri, features, labels)
    except (OSError, MlflowException, ModelVerificationError) as error:
        LOGGER.error("Model verification failed: %s", error)
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Model OK: %d rows scored, hold-out accuracy %.4f.",
                report["rows"], report["holdout_accuracy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
