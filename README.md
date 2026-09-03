# Smart Grid Energy Demand & Tariff Optimization

v1 forecasts **hourly PECO (PJM sub-BA `PE`) demand**, 24h ahead, with peak/stress alerts and PECO Rate R Time-of-Use generation charges (not a made-up tariff).

## What is implemented

- Modular Python package under `src/smart_grid/`
- EIA hourly demand ingest (`parent=PJM`, `subba=PE`) + Open-Meteo weather
- Preprocess / features / frozen **ensemble_mean** (LightGBM + XGBoost)
- MLflow registry (DagsHub if `MLFLOW_TRACKING_URI` is set, else local `mlruns/`)
- FastAPI + Streamlit serving from the local pin `models/production/`

## Freeze the model and register

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train_and_register.py
```

This writes `models/production/` (always) and logs/registers `peco-demand-ensemble` in MLflow (local `mlflow.db`, or DagsHub when `MLFLOW_TRACKING_URI` is set). Official selection scores are the Colab test metrics (WAPE 2.89%, Peak-MAE 287 MW), not a new search.

To use DagsHub, set in `.env`:

```text
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<token>
```

## Serve

```bash
python scripts/run_api.py
python scripts/run_ui.py
```

- API: http://127.0.0.1:8000/docs
- UI: http://127.0.0.1:8501

`POST /forecast` with `{ "grid_id": "PJM_PE", "horizon_hours": 24 }`.

## Streamlit Community Cloud (public UI)

Cloud cannot run FastAPI. The UI calls the same serving code in-process. FastAPI is unchanged for local/Docker.

1. Commit and push these files to `main`.
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → `hamzajawad123/smart-grid-project` → branch `main`.
4. Main file path: `frontend/Overview.py`.
5. Advanced settings → Secrets (TOML). Put your EIA key only:

```toml
EIA_API_KEY = "your-key"
```

Do **not** set `SMART_GRID_API_URL` (that would try a remote API).
6. Deploy. First boot can take several minutes (install + 14-day EIA ingest).
7. Share the `*.streamlit.app` URL. The app sleeps when idle; the first visitor after sleep waits for a restart.

## Refresh data (keeps the live origin current)

```bash
python scripts/refresh_data.py
```

Incremental EIA + weather (last ~2 days), then preprocess, features, and a drift report at `data/processed/drift_report.json`. `/monitoring/drift` exposes the same numbers.

## Daily gated retrain

```bash
python scripts/gated_retrain.py
```

Refits the locked LightGBM+XGBoost ensemble. Writes `models/candidate/`. Copies to `models/production/` **only if** test WAPE improves (Peak-MAE breaks a near-tie).

## Docker

Open Docker Desktop, then:

```bash
docker compose up --build
```

API http://127.0.0.1:8000  ·  UI http://127.0.0.1:8501

Push (after `docker login`):

One Hub image (`smart-grid`). Compose starts API and UI from it with different commands.

```bash
docker compose build
docker push %DOCKERHUB_USER%/smart-grid:latest
```

Feast / Postgres / Redis are not in v1 serving (CSV/Parquet + `models/production/`).

## GitHub Actions

Workflows in `.github/workflows/`: hourly ingest, daily gated retrain, Docker Hub publish. Repo secrets: `EIA_API_KEY`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

## Notebooks (Colab)

- `notebooks/01_smart_grid_eda.ipynb` — EDA on the joined CSV
- `notebooks/02_training.ipynb` — walk-forward training (selection already frozen)

## Data layout

```text
data/raw/          energy and weather as ingested
data/processed/    clean + feature tables
models/production/ frozen ensemble pin
```
