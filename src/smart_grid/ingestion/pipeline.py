"""One-call raw ingest: EIA demand + Open-Meteo weather (historical + live tail)."""

from __future__ import annotations

import logging

import pandas as pd

from smart_grid.config import data_dir, load_params
from smart_grid.ingestion.eia_demand import fetch_demand
from smart_grid.ingestion.open_meteo import fetch_historical_forecast, fetch_live_forecast
from smart_grid.ingestion.raw_store import (
    export_processed,
    load_joined,
    load_joined_csv,
    new_ingest_id,
    save_raw,
)

logger = logging.getLogger(__name__)

ENERGY_STABLE = "peco_hourly_demand.csv"
WEATHER_STABLE = "philadelphia_hourly_weather.csv"


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([old, new], ignore_index=True)
    both["ts_utc"] = pd.to_datetime(both["ts_utc"], utc=True)
    return (
        both.drop_duplicates(subset=["grid_id", "ts_utc"], keep="last")
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )


def _weather_union(historical: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in historical.columns if c in live.columns]
    return _merge(historical, live[cols])


def ingest_raw(
    start: str | None = None,
    end: str | None = None,
    incremental: bool = True,
) -> dict:
    params = load_params()
    raw = data_dir() / "raw"
    energy_path = raw / ENERGY_STABLE
    weather_path = raw / WEATHER_STABLE
    existing_energy = None
    existing_weather = None

    if incremental and energy_path.exists():
        existing_energy = pd.read_csv(energy_path)
        existing_energy["ts_utc"] = pd.to_datetime(existing_energy["ts_utc"], utc=True)
        start = (existing_energy["ts_utc"].max() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        logger.info("Incremental ingest from %s", start)
        if weather_path.exists():
            existing_weather = pd.read_csv(weather_path)
            existing_weather["ts_utc"] = pd.to_datetime(existing_weather["ts_utc"], utc=True)
    elif incremental:
        start = start or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14)).strftime(
            "%Y-%m-%d"
        )
        logger.info("No stable energy file; ingesting last 14 days from %s", start)
    else:
        start = start or params["start_date"]

    ingest_id = new_ingest_id()
    logger.info("Starting raw ingest %s from %s", ingest_id, start)

    energy_new = fetch_demand(start=start, end=end)
    weather_hist = fetch_historical_forecast(start=start, end=end)
    try:
        weather_live = fetch_live_forecast(forecast_days=2, past_days=4, force=True)
        weather_new = _weather_union(weather_hist, weather_live)
    except Exception:
        logger.exception("Live weather merge failed; using historical forecast only")
        weather_new = weather_hist

    energy = _merge(existing_energy, energy_new) if existing_energy is not None else energy_new
    weather = _merge(existing_weather, weather_new) if existing_weather is not None else weather_new

    energy_meta = save_raw("energy", energy_new, ingest_id=ingest_id)
    weather_meta = save_raw("weather", weather_new, ingest_id=ingest_id)
    energy.to_csv(energy_path, index=False)
    weather.to_csv(weather_path, index=False)
    processed_paths = export_processed()
    logger.info(
        "Saved energy=%s weather=%s origin=%s",
        len(energy),
        len(weather),
        energy["ts_utc"].max(),
    )
    return {
        "ingest_id": ingest_id,
        "energy": energy_meta,
        "weather": weather_meta,
        "processed": processed_paths,
        "energy_rows": int(len(energy)),
        "weather_rows": int(len(weather)),
        "last_demand_utc": pd.Timestamp(energy["ts_utc"].max()).isoformat(),
    }


def load_eda_frame(start: str | None = None, end: str | None = None, refresh: bool = False):
    if not refresh:
        try:
            return load_joined_csv()
        except FileNotFoundError:
            pass
        try:
            return load_joined()
        except FileNotFoundError:
            pass
    ingest_raw(start=start, end=end)
    return load_joined_csv()
