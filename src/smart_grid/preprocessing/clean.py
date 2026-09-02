"""Clean joined hourly PECO data. No clipping of real peaks. UTC hourly index."""

from __future__ import annotations

import logging

import pandas as pd

from smart_grid.config import data_dir, load_params

logger = logging.getLogger(__name__)

WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
]

TARGET = "demand_mw"


def _complete_hourly_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
    frame = frame.sort_values("ts_utc").drop_duplicates(subset=["grid_id", "ts_utc"], keep="first")
    full = pd.date_range(frame["ts_utc"].min(), frame["ts_utc"].max(), freq="h", tz="UTC")
    frame = frame.set_index("ts_utc").reindex(full)
    frame.index.name = "ts_utc"
    return frame


def fill_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill DST-sized holes (EDA: 23–25h in Mar/Nov). Demand uses week-ago, then linear."""
    out = frame.copy()
    out["was_imputed"] = out[TARGET].isna()

    for col in ["grid_id", "source", "ingest_id"]:
        if col in out.columns:
            out[col] = out[col].ffill().bfill()

    demand = out[TARGET]
    demand = demand.fillna(demand.shift(168)).fillna(demand.shift(-168))
    demand = demand.interpolate(method="time", limit=26)
    out[TARGET] = demand

    weather = [c for c in WEATHER_COLS if c in out.columns]
    out[weather] = out[weather].interpolate(method="time", limit=26)
    out[weather] = out[weather].ffill().bfill()
    return out


def add_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Chronological split only. Test = last ~12 months; val = 12 months before that."""
    params = load_params()
    split_cfg = params.get("split", {})
    val_start = pd.Timestamp(split_cfg.get("val_start", "2024-08-31"), tz="UTC")
    test_start = pd.Timestamp(split_cfg.get("test_start", "2025-08-31"), tz="UTC")
    out = frame.copy()
    ts = out.index if isinstance(out.index, pd.DatetimeIndex) else pd.to_datetime(out["ts_utc"], utc=True)
    labels = pd.Series("train", index=out.index)
    labels = labels.mask(ts >= val_start, "val")
    labels = labels.mask(ts >= test_start, "test")
    out["split"] = labels
    return out


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    out = _complete_hourly_index(frame)
    out = fill_gaps(out)
    out = add_split(out)
    out = out.reset_index()
    still_missing = int(out[TARGET].isna().sum())
    if still_missing:
        raise ValueError(f"Demand still has {still_missing} missing hours after fill.")
    logger.info(
        "Preprocessed rows=%s imputed=%s splits=%s",
        len(out),
        int(out["was_imputed"].sum()),
        out["split"].value_counts().to_dict(),
    )
    return out


def run_preprocess() -> dict:
    raw_joined = data_dir() / "processed" / "peco_demand_weather_joined.csv"
    if not raw_joined.exists():
        raise FileNotFoundError(f"Missing {raw_joined}. Ingest first.")
    frame = pd.read_csv(raw_joined)
    cleaned = preprocess(frame)
    out_csv = data_dir() / "processed" / "peco_clean.csv"
    out_parquet = data_dir() / "processed" / "peco_clean.parquet"
    cleaned.to_csv(out_csv, index=False)
    cleaned.to_parquet(out_parquet, index=False)
    return {
        "csv": str(out_csv),
        "parquet": str(out_parquet),
        "rows": int(len(cleaned)),
        "imputed": int(cleaned["was_imputed"].sum()),
        "splits": cleaned["split"].value_counts().to_dict(),
    }
