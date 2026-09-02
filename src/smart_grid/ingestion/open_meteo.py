"""Open-Meteo historical forecast weather (training-aligned)."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from smart_grid.config import load_params
from smart_grid.utils.http import get_json

logger = logging.getLogger(__name__)


def _date_chunks(start: date, end: date, days: int = 180) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def fetch_historical_forecast(
    start: str | date,
    end: str | date | None = None,
) -> pd.DataFrame:
    params = load_params()
    weather = params["weather"]
    grid_id = params["grid_id"]
    start_d = pd.Timestamp(start).date()
    end_d = (
        pd.Timestamp(end).date()
        if end is not None
        else (pd.Timestamp.now(tz="UTC").date() - timedelta(days=1))
    )

    frames: list[pd.DataFrame] = []
    hourly = ",".join(weather["hourly"])
    for chunk_start, chunk_end in _date_chunks(start_d, end_d):
        query = {
            "latitude": weather["latitude"],
            "longitude": weather["longitude"],
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": hourly,
            "timezone": "UTC",
        }
        payload = get_json(weather["historical_forecast_url"], params=query)
        hourly_block = payload.get("hourly") or {}
        times = hourly_block.get("time") or []
        if not times:
            logger.warning("No weather rows for %s to %s", chunk_start, chunk_end)
            continue
        chunk = pd.DataFrame({"ts_utc": pd.to_datetime(times, utc=True)})
        for col in weather["hourly"]:
            chunk[col] = hourly_block.get(col)
        frames.append(chunk)
        logger.info("Weather chunk %s → %s (%s rows)", chunk_start, chunk_end, len(chunk))
        time.sleep(0.4)

    if not frames:
        raise RuntimeError("Open-Meteo returned no historical forecast rows.")

    frame = pd.concat(frames, ignore_index=True)
    frame["grid_id"] = grid_id
    frame["source"] = "open_meteo_historical_forecast"
    frame = (
        frame.drop_duplicates(subset=["grid_id", "ts_utc"])
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )
    return frame


def fetch_live_forecast(
    forecast_days: int = 3, past_days: int = 2, force: bool = False
) -> pd.DataFrame:
    """Open-Meteo Forecast API (inference). Cached for weather.cache_hours."""
    from smart_grid.config import data_dir

    params = load_params()
    weather = params["weather"]
    grid_id = params["grid_id"]
    cache_hours = float(weather.get("cache_hours", 12))
    cache_csv = data_dir() / "processed" / "open_meteo_forecast_cache.csv"
    stamp_path = data_dir() / "processed" / "open_meteo_forecast_cache.stamp"

    if not force and cache_csv.exists() and stamp_path.exists():
        fetched = pd.Timestamp(stamp_path.read_text(encoding="utf-8").strip())
        if fetched.tzinfo is None:
            fetched = fetched.tz_localize("UTC")
        age_h = (pd.Timestamp.now(tz="UTC") - fetched).total_seconds() / 3600
        if age_h < cache_hours:
            frame = pd.read_csv(cache_csv)
            frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
            logger.info("Weather forecast cache hit (%.1fh old)", age_h)
            return frame

    query = {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "hourly": ",".join(weather["hourly"]),
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "past_days": past_days,
    }
    payload = get_json(weather["forecast_url"], params=query)
    hourly_block = payload.get("hourly") or {}
    times = hourly_block.get("time") or []
    if not times:
        raise RuntimeError("Open-Meteo forecast returned no hourly rows.")
    frame = pd.DataFrame({"ts_utc": pd.to_datetime(times, utc=True)})
    for col in weather["hourly"]:
        frame[col] = hourly_block.get(col)
    frame["grid_id"] = grid_id
    frame["source"] = "open_meteo_forecast"
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_csv, index=False)
    stamp_path.write_text(pd.Timestamp.now(tz="UTC").isoformat(), encoding="utf-8")
    logger.info("Fetched live weather forecast (%s rows)", len(frame))
    return frame
