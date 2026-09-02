"""FastAPI serving layer. Loads the local production pin; MLflow is the registry."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from smart_grid.config import api_key, load_params, mlflow_tracking_uri, models_dir
from smart_grid.models.train import load_production_pin
from smart_grid.serving.forecast import drift_summary, historical, last_forecast, load_history, run_forecast


class ForecastRequest(BaseModel):
    grid_id: str = "PJM_PE"
    horizon_hours: int = Field(24, ge=1, le=24)
    include_shap: bool = True


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict | None = None


def _check_key(x_api_key: str | None) -> None:
    expected = api_key()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Invalid or missing X-API-Key"})


def _http_error(status: int, code: str, message: str, details: dict | None = None) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, "details": details or {}})


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Grid PECO Demand API",
        version="0.1.0",
        description="24h-ahead PECO (PJM_PE) demand forecast. Tariff uses PECO Rate R TOU generation charges.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        params = load_params()
        pin_ok = (models_dir() / "ensemble.joblib").exists()
        last_obs = None
        hours_stale = None
        try:
            hist = load_history()
            ts = hist["ts_utc"].max()
            last_obs = pd_ts(ts)
            hours_stale = round(
                (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600,
                2,
            )
            data_ok = True
        except FileNotFoundError:
            data_ok = False
        status = "ok" if pin_ok and data_ok else "degraded"
        return {
            "status": status,
            "grid_id": params["grid_id"],
            "model_loaded": pin_ok,
            "data_loaded": data_ok,
            "last_observation_utc": last_obs,
            "hours_stale": hours_stale,
            "mlflow_tracking_uri_set": bool(mlflow_tracking_uri()),
        }

    @app.get("/model/info")
    def model_info(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        try:
            pin = load_production_pin()
        except FileNotFoundError as exc:
            raise _http_error(503, "model_missing", str(exc)) from exc
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

    @app.post("/forecast")
    def post_forecast(body: ForecastRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        try:
            return run_forecast(
                grid_id=body.grid_id,
                horizon_hours=body.horizon_hours,
                include_shap=body.include_shap,
            )
        except FileNotFoundError as exc:
            raise _http_error(503, "not_ready", str(exc)) from exc
        except ValueError as exc:
            raise _http_error(400, "bad_request", str(exc)) from exc
        except Exception as exc:
            raise _http_error(502, "forecast_failed", str(exc)) from exc

    @app.get("/forecast/{grid_id}")
    def get_forecast(grid_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        stored = last_forecast()
        if stored is None:
            raise _http_error(404, "no_forecast", "No forecast stored yet. POST /forecast first.")
        if stored.get("grid_id") != grid_id:
            raise _http_error(404, "no_forecast", f"No stored forecast for {grid_id}.")
        return stored

    @app.get("/historical/{grid_id}")
    def get_historical(
        grid_id: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = Query(168, ge=1, le=5000),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        _check_key(x_api_key)
        try:
            return historical(grid_id, start=start, end=end, limit=limit)
        except FileNotFoundError as exc:
            raise _http_error(503, "not_ready", str(exc)) from exc
        except ValueError as exc:
            raise _http_error(400, "bad_request", str(exc)) from exc

    @app.get("/peak-alerts")
    def peak_alerts(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        stored = last_forecast()
        if stored is None:
            raise _http_error(404, "no_forecast", "No forecast stored yet. POST /forecast first.")
        return {
            "grid_id": stored["grid_id"],
            "label": "Capacity Stress Index (statistical P90/P95, not physical overload)",
            "alerts": stored.get("peak_alerts", []),
        }

    @app.get("/tariff-recommendation")
    def tariff_recommendation(
        grid_id: str = Query("PJM_PE"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        _check_key(x_api_key)
        stored = last_forecast()
        if stored is None:
            raise _http_error(404, "no_forecast", "No forecast stored yet. POST /forecast first.")
        if stored.get("grid_id") != grid_id:
            raise _http_error(400, "bad_request", f"Only {stored.get('grid_id')} has a stored forecast.")
        return stored["tariff"]

    @app.get("/monitoring/drift")
    def monitoring_drift(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        try:
            return drift_summary()
        except FileNotFoundError as exc:
            raise _http_error(503, "not_ready", str(exc)) from exc

    @app.get("/explainability")
    def explainability(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        try:
            pin = load_production_pin()
        except FileNotFoundError as exc:
            raise _http_error(503, "model_missing", str(exc)) from exc
        shap_global = pin["metadata"].get("shap_global")
        if not shap_global:
            raise _http_error(404, "shap_missing", "Run python scripts/compute_shap.py")
        return {
            "method": "0.5 * TreeExplainer(LightGBM) + 0.5 * TreeExplainer(XGBoost)",
            "unit": "MW (mean |SHAP| on a train sample)",
            "shap_global": shap_global,
        }

    return app


def pd_ts(value) -> str:
    return value.isoformat()


app = create_app()
