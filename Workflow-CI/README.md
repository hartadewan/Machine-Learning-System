# Workflow-CI — Credit Scoring (Kriteria 3)

On every push to `main` (or a manual run), GitHub Actions retrains the
credit-scoring model through an **MLflow Project**, checks that the logged model
loads and predicts, stores the model in a **GitHub Release** of this repository,
and pushes a serving image built with **`mlflow models build-docker`** to
**Docker Hub**.

## Layout and names used in the Dicoding guide

| Guide (PDF) | This repository | Note |
| --- | --- | --- |
| `.workflow` | `.github/workflows/ci.yml` | GitHub only runs workflows from this folder |
| `MLProject` (folder) | `MLProject/` | |
| `MLProject` (file) | `MLProject/MLproject` | MLflow reads this name case-insensitively |
| `modelling.py`, `conda.yaml` | `MLProject/modelling.py`, `MLProject/conda.yaml` | |
| `namadataset_preprocessing` | `MLProject/namadataset_preprocessing/` | `train.csv`, `test.csv` (800/200 rows, 61 features + `credit_risk`) |
| Tautan ke Docker Hub | `MLProject/DockerHub.txt` | **Replace `<DOCKERHUB_USERNAME>`** |
| file tambahan | `ci/` | small CI helpers (env export, run lookup, model check, manifest, serving check) |

## One-time setup

1. Create a **public** GitHub repository named `Workflow-CI` and push this tree to `main`.
2. Add repository secrets (*Settings → Secrets and variables → Actions*):

| Secret | Value | Minimum scope |
| --- | --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub namespace (lowercase) | — |
| `DOCKERHUB_TOKEN` | Docker Hub personal access token | **Read & Write** (Delete not needed) |

3. `GITHUB_TOKEN` needs no secret. The workflow declares `permissions: {}` and
   grants the job only `contents: write` (create the release, its tag and assets).
   If an organisation policy caps tokens at read-only, allow write for this repo.
4. After the first push, make the Docker Hub repository `credit-scoring-model`
   **public** so reviewers can pull it, and put its URL in `MLProject/DockerHub.txt`.

## Triggers

- `push` to `main`
- `workflow_dispatch` (Actions tab → *CI - Train and Ship Credit Scoring Model* → *Run workflow*)

## What one run does

| Step (guide name) | What happens |
| --- | --- |
| Set up Python 3.12.7 | `actions/setup-python@v7` |
| Check Env | data files and target column present, secrets set — fails before training otherwise |
| Install dependencies | installs the pins in `conda.yaml`; fails if Python ≠ the `python=` pin |
| Set MLflow Tracking URI | fresh file store in the runner's temp dir (never committed or published) |
| Run mlflow project | `mlflow run MLProject --env-manager=local` |
| Get latest MLflow run_id | reads the run back from MLflow; requires FINISHED, one pyfunc model, matching commit |
| *Verify logged model* | loads `runs:/<id>/model` and scores all 200 hold-out rows |
| Install Python dependencies | isolated venv from the logged model's own `requirements.txt` |
| *Package model artifacts* | `.tar.gz` of the model folder, re-loaded in that venv; writes `manifest.json`, `SHA256SUMS` |
| Upload to GitHub | draft release `model-<sha7>-<runid12>` with the three files (replaces "Upload to Google Drive") |
| Build Docker Model | `mlflow models build-docker --env-manager local` (base `python:3.12.7-slim`) |
| *Smoke-test Docker image* | runs the container, `GET /ping`, `POST /invocations` with 5 hold-out rows |
| Log in / Tag / Push Docker Image | tags `<sha7>-<runid12>` (immutable) and `latest`; digest read back from the registry |
| *Publish GitHub Release* | adds the digest to `manifest.json`, publishes the release, checks its assets |

If any step after the draft fails, the draft is deleted, so only fully
successful runs leave a release. A copy of the outputs is also kept as an
Actions artifact for 14 days (convenience only; the release is the stored copy).
The checkout step shows as `actions/checkout@v7`: the guide's screenshot shows
v3, which targets the retired Node 16 runtime.

## Where the stored model is

*Releases → `model-<version>`*, where `<version>` = `<first 7 chars of commit SHA>-<first 12 chars of MLflow run ID>`:

- `credit-scoring-model-<version>.tar.gz` — `model/` with `MLmodel`, `model.pkl`,
  `conda.yaml`, `python_env.yaml`, `requirements.txt`
- `manifest.json` — MLflow run ID, model URI, metrics, params, hold-out check,
  commit SHA, workflow run ID/URL, archive SHA-256, image tags and digest
- `SHA256SUMS`

`mlflow.model_location` in the manifest is where MLflow stored the model on the
runner during the job; the durable copy is `model_archive.download_url`.

```bash
gh release download model-<version> --repo <owner>/Workflow-CI
sha256sum -c SHA256SUMS && tar -xzf credit-scoring-model-<version>.tar.gz
mlflow models serve -m model --env-manager local -p 5002
```

## Pull and serve the Docker Hub image

```bash
docker pull docker.io/<DOCKERHUB_USERNAME>/credit-scoring-model:<version>   # or :latest
docker run --rm -p 5005:8080 docker.io/<DOCKERHUB_USERNAME>/credit-scoring-model:<version>
# in another terminal (needs pandas):
python ci/smoke_test_serving.py --url http://127.0.0.1:5005 \
  --data MLProject/namadataset_preprocessing/test.csv --target credit_risk
```

`POST /invocations` takes MLflow's `{"dataframe_split": {"columns": [...], "data": [[...]]}}`
with the 61 preprocessed feature columns and returns `{"predictions": [0, 1, ...]}`
(1 = bad credit risk).

## Run the MLflow Project locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # must be exactly 3.12.7 (checked)
pip install -r ci/requirements.txt
python ci/export_pip_requirements.py MLProject/conda.yaml --output /tmp/reqs.txt
pip install -r /tmp/reqs.txt
mlflow run MLProject --env-manager local                     # runs land in ./mlruns
# or, with conda installed:  mlflow run MLProject
```

`python MLProject/modelling.py` still logs to `http://127.0.0.1:5000/` (Kriteria 2
behaviour) unless `MLFLOW_TRACKING_URI` is set.

## Data provenance

`namadataset_preprocessing/` comes from the Kriteria 1 preprocessing (stratified
80/20 split, `random_state=42`, StandardScaler + OneHotEncoder). SHA-256:

```
9bf3a981d19766c03ee581e0c47b1a303a267698f0396a2ea39e98eb96e009a7  train.csv
2b3a32c7e6166d180af4171e71a82e723300139ad3f5b2d07ed20a3e4ef3f54f  test.csv
```

If your Kriteria 1 output has different checksums, replace both files with yours.
