"""HTTP-free handlers shared by FastAPI and Streamlit Community Cloud."""

from __future__ import annotations

import logging
from urllib.parse import unquote

import pandas as pd

from smart_grid.config import load_params, mlflow_tracking_uri, models_dir
from smart_grid.models.train import load_production_pin
from smart_grid.serving.forecast import drift_summary, historical, last_forecast, load_history, run_forecast

logger = logging.getLogger(__name__)


class ServingError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def _iso(value) -> str:
    return pd.Timestamp(value).isoformat()


def ensure_serving_tables() -> None:
    """If processed demand is missing (Cloud clone), ingest the last 14 days."""
    try:
        load_history()
        return
    except FileNotFoundError:
        pass
    logger.info("No peco_clean.csv; ingesting a 14-day window for serving")
    from smart_grid.ingestion.pipeline import ingest_raw
    from smart_grid.preprocessing.clean import run_preprocess

    ingest_raw(incremental=True)
    run_preprocess()
    load_history()


def health() -> dict:
    params = load_params()
    pin_ok = (models_dir() / "ensemble.joblib").exists()
    last_obs = None
    hours_stale = None
    try:
        hist = load_history()
        ts = hist["ts_utc"].max()
        last_obs = _iso(ts)
        hours_stale = round((pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600, 2)
        data_ok = True
    except FileNotFoundError:
        data_ok = False
    return {
        "status": "ok" if pin_ok and data_ok else "degraded",
        "grid_id": params["grid_id"],
        "model_loaded": pin_ok,
        "data_loaded": data_ok,
        "last_observation_utc": last_obs,
        "hours_stale": hours_stale,
        "mlflow_tracking_uri_set": bool(mlflow_tracking_uri()),
    }


def model_info() -> dict:
    try:
        pin = load_production_pin()
    except FileNotFoundError as exc:
        raise ServingError(503, "model_missing", str(exc)) from exc
    meta = pin["metadata"]
    return {
        "name": meta["winner"],
        "trained_at": meta["trained_at"],
        "grid_id": meta["grid_id"],
        "horizon_hours": meta["horizon_hours"],
        "thresholds_mw": meta["thresholds_mw"],
        "refit_metrics": meta["refit_metrics"],
        "colab_selection_metrics": meta["colab_selection_metrics"]["ensemble_mean"],
        "n_train": meta["n_train"],
        "feature_importance_lgbm": meta.get("feature_importance_lgbm", {}),
        "shap_global": meta.get("shap_global"),
    }


def explainability() -> dict:
    try:
        pin = load_production_pin()
    except FileNotFoundError as exc:
        raise ServingError(503, "model_missing", str(exc)) from exc
    shap_global = pin["metadata"].get("shap_global")
    if not shap_global:
        raise ServingError(404, "shap_missing", "Run python scripts/compute_shap.py")
    return {
        "method": "0.5 * TreeExplainer(LightGBM) + 0.5 * TreeExplainer(XGBoost)",
        "unit": "MW (mean |SHAP| on a train sample)",
        "shap_global": shap_global,
    }


def post_forecast(body: dict | None = None) -> dict:
    body = body or {}
    try:
        return run_forecast(
            grid_id=body.get("grid_id"),
            horizon_hours=body.get("horizon_hours"),
            include_shap=bool(body.get("include_shap", True)),
        )
    except FileNotFoundError as exc:
        raise ServingError(503, "not_ready", str(exc)) from exc
    except ValueError as exc:
        raise ServingError(400, "bad_request", str(exc)) from exc
    except Exception as exc:
        raise ServingError(502, "forecast_failed", str(exc)) from exc


def get_forecast(grid_id: str, *, create_if_missing: bool = False) -> dict:
    stored = last_forecast()
    if stored is None:
        if create_if_missing:
            return post_forecast({"grid_id": grid_id, "horizon_hours": 24, "include_shap": False})
        raise ServingError(404, "no_forecast", "No forecast stored yet. POST /forecast first.")
    if stored.get("grid_id") != grid_id:
        raise ServingError(404, "no_forecast", f"No stored forecast for {grid_id}.")
    return stored


def get_historical(grid_id: str, params: dict | None = None) -> dict:
    params = params or {}
    try:
        return historical(
            grid_id,
            start=params.get("start"),
            end=params.get("end"),
            limit=int(params.get("limit", 168)),
        )
    except FileNotFoundError as exc:
        raise ServingError(503, "not_ready", str(exc)) from exc
    except ValueError as exc:
        raise ServingError(400, "bad_request", str(exc)) from exc


def peak_alerts() -> dict:
    stored = last_forecast()
    if stored is None:
        raise ServingError(404, "no_forecast", "No forecast stored yet. POST /forecast first.")
    return {
        "grid_id": stored["grid_id"],
        "label": "Capacity Stress Index (statistical P90/P95, not physical overload)",
        "alerts": stored.get("peak_alerts", []),
    }


def tariff_recommendation(grid_id: str) -> dict:
    stored = last_forecast()
    if stored is None:
        raise ServingError(404, "no_forecast", "No forecast stored yet. POST /forecast first.")
    if stored.get("grid_id") != grid_id:
        raise ServingError(400, "bad_request", f"Only {stored.get('grid_id')} has a stored forecast.")
    return stored["tariff"]


def monitoring_drift() -> dict:
    try:
        return drift_summary()
    except FileNotFoundError as exc:
        raise ServingError(503, "not_ready", str(exc)) from exc


def call(
    method: str,
    path: str,
    params: dict | None = None,
    body: dict | None = None,
    *,
    bootstrap: bool = False,
    create_forecast_if_missing: bool = False,
) -> dict:
    method = method.upper()
    path = unquote(path.split("?", 1)[0]).rstrip("/") or "/"
    params = params or {}
    if bootstrap and path not in {"/health", "/model/info", "/explainability"}:
        try:
            ensure_serving_tables()
        except Exception as exc:
            raise ServingError(503, "not_ready", str(exc)) from exc

    if method == "GET" and path == "/health":
        return health()
    if method == "GET" and path == "/model/info":
        return model_info()
    if method == "GET" and path == "/explainability":
        return explainability()
    if method == "POST" and path == "/forecast":
        return post_forecast(body)
    if method == "GET" and path.startswith("/forecast/"):
        return get_forecast(path.rsplit("/", 1)[-1], create_if_missing=create_forecast_if_missing)
    if method == "GET" and path.startswith("/historical/"):
        return get_historical(path.rsplit("/", 1)[-1], params)
    if method == "GET" and path == "/peak-alerts":
        if create_forecast_if_missing and last_forecast() is None:
            get_forecast(load_params()["grid_id"], create_if_missing=True)
        return peak_alerts()
    if method == "GET" and path == "/tariff-recommendation":
        grid_id = str(params.get("grid_id") or load_params()["grid_id"])
        if create_forecast_if_missing and last_forecast() is None:
            get_forecast(grid_id, create_if_missing=True)
        return tariff_recommendation(grid_id)
    if method == "GET" and path == "/monitoring/drift":
        return monitoring_drift()
    raise ServingError(404, "not_found", f"No handler for {method} {path}")
