"""Load history, live weather, produce a 24h ensemble forecast."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smart_grid.config import data_dir, load_params
from smart_grid.features.build import FEATURE_COLS, TARGET
from smart_grid.ingestion.open_meteo import fetch_live_forecast
from smart_grid.models.train import load_production_pin
from smart_grid.serving.bands import stress_band
from smart_grid.serving.features import horizon_features
from smart_grid.tariff.tou import annotate_hours, recommend


def _clean_path() -> Path:
    return data_dir() / "processed" / "peco_clean.csv"


def _forecast_store() -> Path:
    return data_dir() / "processed" / "last_forecast.json"


def load_history() -> pd.DataFrame:
    path = _clean_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ingest + preprocess first.")
    frame = pd.read_csv(path)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame.sort_values("ts_utc")


def run_forecast(grid_id: str | None = None, horizon_hours: int | None = None, include_shap: bool = True) -> dict:
    params = load_params()
    expected = params["grid_id"]
    grid_id = grid_id or expected
    if grid_id != expected:
        raise ValueError(f"Only {expected} is served in v1 (got {grid_id}).")
    horizon_hours = int(horizon_hours or params.get("horizon_hours", 24))
    if horizon_hours < 1 or horizon_hours > 24:
        raise ValueError("horizon_hours must be 1–24 (direct 24h-ahead model).")

    pin = load_production_pin()
    ensemble = pin["ensemble"]
    meta = pin["metadata"]
    history = load_history()
    last_obs = history["ts_utc"].max()
    age_hours = (pd.Timestamp.now(tz="UTC") - last_obs).total_seconds() / 3600
    past_days = max(2, min(int(age_hours / 24) + 2, 14))
    weather = fetch_live_forecast(forecast_days=8, past_days=past_days)
    feats = horizon_features(history, weather, horizon_hours=horizon_hours)
    preds = ensemble.predict(feats[FEATURE_COLS])
    thresholds = meta["thresholds_mw"]
    hours = []
    for ts, mw, temp in zip(feats["ts_utc"], preds, feats["temperature_2m"]):
        hours.append(
            {
                "ts_utc": pd.Timestamp(ts).isoformat(),
                "demand_mw": round(float(mw), 2),
                "temperature_2m": round(float(temp), 2) if pd.notna(temp) else None,
                "band": stress_band(
                    float(mw),
                    thresholds["p75"],
                    thresholds["p90"],
                    thresholds["p95"],
                ),
            }
        )
    hours = annotate_hours(hours)
    if include_shap and pin.get("lgbm") is not None and pin.get("xgb") is not None:
        from smart_grid.models.explain import local_rows

        weight = float(meta.get("ensemble_weight_lgbm", 0.5))
        shap_hours = local_rows(pin["lgbm"], pin["xgb"], feats[FEATURE_COLS], weight_lgbm=weight)
        for row, shap_row in zip(hours, shap_hours):
            row["shap"] = shap_row
    payload = {
        "grid_id": grid_id,
        "horizon_hours": horizon_hours,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "origin_ts_utc": pd.Timestamp(last_obs).isoformat(),
        "model": {
            "name": meta["winner"],
            "trained_at": meta["trained_at"],
            "thresholds_mw": thresholds,
        },
        "hours": hours,
        "tariff": recommend(hours),
        "peak_alerts": [h for h in hours if h["band"] in {"high", "critical"}],
    }
    _forecast_store().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def last_forecast() -> dict | None:
    path = _forecast_store()
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    hours = payload.get("hours") or []
    if hours:
        payload["hours"] = annotate_hours(hours)
        payload["tariff"] = recommend(payload["hours"])
        payload["peak_alerts"] = [h for h in payload["hours"] if h.get("band") in {"high", "critical"}]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def historical(grid_id: str, start: str | None = None, end: str | None = None, limit: int = 168) -> dict:
    params = load_params()
    if grid_id != params["grid_id"]:
        raise ValueError(f"Only {params['grid_id']} is served in v1.")
    frame = load_history()
    if start:
        frame = frame[frame["ts_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        frame = frame[frame["ts_utc"] <= pd.Timestamp(end, tz="UTC")]
    frame = frame.tail(int(limit))
    keep = ["ts_utc", "grid_id", TARGET, "temperature_2m", "split", "was_imputed"]
    keep = [c for c in keep if c in frame.columns]
    rows = []
    for _, row in frame[keep].iterrows():
        item = {c: row[c] for c in keep}
        item["ts_utc"] = pd.Timestamp(item["ts_utc"]).isoformat()
        item[TARGET] = float(item[TARGET])
        rows.append(item)
    return {"grid_id": grid_id, "rows": rows}


def drift_summary() -> dict:
    """Input drift vs train (PSI + z). Writes data/processed/drift_report.json."""
    from smart_grid.serving.monitoring import drift_summary as _drift

    return _drift()
