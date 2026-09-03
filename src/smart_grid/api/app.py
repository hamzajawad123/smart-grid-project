"""FastAPI serving layer. Loads the local production pin; MLflow is the registry."""

from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from smart_grid.config import api_key
from smart_grid.serving.dispatch import ServingError, call


class ForecastRequest(BaseModel):
    grid_id: str = "PJM_PE"
    horizon_hours: int = Field(24, ge=1, le=24)
    include_shap: bool = True


def _check_key(x_api_key: str | None) -> None:
    expected = api_key()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Invalid or missing X-API-Key"})


def _run(method: str, path: str, params: dict | None = None, body: dict | None = None):
    try:
        return call(method, path, params=params, body=body, bootstrap=False)
    except ServingError as exc:
        raise HTTPException(
            status_code=exc.status,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


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
        return _run("GET", "/health")

    @app.get("/model/info")
    def model_info(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        return _run("GET", "/model/info")

    @app.post("/forecast")
    def post_forecast(body: ForecastRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        return _run("POST", "/forecast", body=body.model_dump())

    @app.get("/forecast/{grid_id}")
    def get_forecast(grid_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        return _run("GET", f"/forecast/{grid_id}")

    @app.get("/historical/{grid_id}")
    def get_historical(
        grid_id: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = Query(168, ge=1, le=5000),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        _check_key(x_api_key)
        params = {"limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return _run("GET", f"/historical/{grid_id}", params=params)

    @app.get("/peak-alerts")
    def peak_alerts(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        return _run("GET", "/peak-alerts")

    @app.get("/tariff-recommendation")
    def tariff_recommendation(
        grid_id: str = Query("PJM_PE"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        _check_key(x_api_key)
        return _run("GET", "/tariff-recommendation", params={"grid_id": grid_id})

    @app.get("/monitoring/drift")
    def monitoring_drift(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        return _run("GET", "/monitoring/drift")

    @app.get("/explainability")
    def explainability(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _check_key(x_api_key)
        return _run("GET", "/explainability")

    return app


app = create_app()
