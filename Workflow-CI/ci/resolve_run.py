from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger("resolve_run")

LOG_MODEL_HISTORY_TAG = "mlflow.log-model.history"
GIT_COMMIT_TAG = "mlflow.source.git.commit"
PYFUNC_FLAVOR = "python_function"


class RunResolutionError(RuntimeError):
 """Raised when the CI run cannot be resolved unambiguously."""

def find_single_run(client: MlflowClient, experiment_name: str) -> Run:

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RunResolutionError(f"Experiment '{experiment_name}' does not exist; "
                                 "did `mlflow run` use a different experiment?")
    runs = client.search_runs([experiment.experiment_id],
                              order_by=["attributes.start_time DESC"], max_results=2)
    if len(runs) != 1:
        raise RunResolutionError(f"Expected exactly 1 run in '{experiment_name}', "
                                 f"found {len(runs)}. The CI tracking store must be fresh.")
    run = runs[0]
    if run.info.status != "FINISHED":
        raise RunResolutionError(f"Run {run.info.run_id} ended with status {run.info.status}.")
    return run


def check_commit(run: Run, expected_sha: str | None) -> str | None:

    recorded = run.data.tags.get(GIT_COMMIT_TAG)
    if expected_sha and recorded and recorded != expected_sha:
        raise RunResolutionError(f"Run commit {recorded} != workflow commit {expected_sha}.")
    if expected_sha and not recorded:
        LOGGER.warning("Run has no %s tag; relying on the single-run check.", GIT_COMMIT_TAG)
    return recorded


def model_artifact_path(client: MlflowClient, run: Run) -> str:

    history = json.loads(run.data.tags.get(LOG_MODEL_HISTORY_TAG, "[]"))
    paths = sorted({entry["artifact_path"] for entry in history
                    if PYFUNC_FLAVOR in entry.get("flavors", {})})
    if len(paths) != 1:
        raise RunResolutionError(f"Expected one logged pyfunc model, found {paths or 'none'}.")
    artifact_path = paths[0]
    files = {Path(item.path).name for item in client.list_artifacts(run.info.run_id, artifact_path)}
    if "MLmodel" not in files:
        raise RunResolutionError(f"No MLmodel file under artifact path '{artifact_path}'.")
    return artifact_path


def write_step_outputs(values: dict[str, str]) -> None:
  
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        LOGGER.info("GITHUB_OUTPUT not set; skipping step outputs.")
        return
    with open(output_file, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")

FALLBACK_DESCRIPTION = "Resolve the single MLflow run produced by this CI build."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(__doc__ or FALLBACK_DESCRIPTION).splitlines()[0])
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--commit-sha", default=None,
                        help="Commit the workflow is building (GITHUB_SHA).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the run facts as JSON.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    client = MlflowClient()
    try:
        run = find_single_run(client, args.experiment_name)
        commit = check_commit(run, args.commit_sha)
        artifact_path = model_artifact_path(client, run)
    except (MlflowException, RunResolutionError, json.JSONDecodeError) as error:
        LOGGER.error("%s", error)
        return 1

    info = run.info
    facts = {
        "run_id": info.run_id,
        "run_name": info.run_name,
        "experiment_id": info.experiment_id,
        "status": info.status,
        "start_time_ms": info.start_time,
        "end_time_ms": info.end_time,
        "git_commit": commit,
        "artifact_uri": info.artifact_uri,
        "model_artifact_path": artifact_path,
        "model_uri": f"runs:/{info.run_id}/{artifact_path}",
        "model_location": f"{info.artifact_uri.rstrip('/')}/{artifact_path}",
        "metrics": dict(sorted(run.data.metrics.items())),
        "params": dict(sorted(run.data.params.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    write_step_outputs({key: str(facts[key]) for key in
                        ("run_id", "model_uri", "model_artifact_path", "model_location")})
    LOGGER.info("Run %s FINISHED; model at %s (%d metrics).",
                info.run_id, facts["model_location"], len(facts["metrics"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
