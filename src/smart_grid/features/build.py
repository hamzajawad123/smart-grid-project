"""Leakage-safe 24h-ahead features. No lag_1 (that would be 1-hour-ahead)."""

from __future__ import annotations

import logging

import holidays
import numpy as np
import pandas as pd

from smart_grid.config import data_dir

logger = logging.getLogger(__name__)

TARGET = "demand_mw"
LAGS = (24, 48, 72, 168)
ROLL_WINDOWS = (24, 168)
HDD_BASE_C = 18.0

FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
    "hdd",
    "cdd",
    "temp_change_24",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "roll_mean_24",
    "roll_std_24",
    "roll_min_24",
    "roll_max_24",
    "roll_mean_168",
    "was_imputed",
]


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["ts_utc"], utc=True).dt.tz_convert("America/New_York")
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    years = range(int(ts.dt.year.min()), int(ts.dt.year.max()) + 1)
    us_holidays = holidays.country_holidays("US", years=years)
    out["is_holiday"] = ts.dt.date.map(lambda d: int(d in us_holidays))
    return out


def add_weather_derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    temp = out["temperature_2m"]
    out["hdd"] = (HDD_BASE_C - temp).clip(lower=0)
    out["cdd"] = (temp - HDD_BASE_C).clip(lower=0)
    # 24h change uses only past temperature (known for a 24h-ahead forecast).
    out["temp_change_24"] = temp - temp.shift(24)
    return out


def add_demand_lags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    y = out[TARGET]
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    shifted = y.shift(24)
    out["roll_mean_24"] = shifted.rolling(24, min_periods=24).mean()
    out["roll_std_24"] = shifted.rolling(24, min_periods=24).std()
    out["roll_min_24"] = shifted.rolling(24, min_periods=24).min()
    out["roll_max_24"] = shifted.rolling(24, min_periods=24).max()
    out["roll_mean_168"] = shifted.rolling(168, min_periods=168).mean()
    return out


def build_features(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    out = frame.sort_values("ts_utc").copy()
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    out = add_calendar(out)
    out = add_weather_derived(out)
    out = add_demand_lags(out)
    keep = ["ts_utc", "grid_id", TARGET, "split", *FEATURE_COLS]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]
    if require_target:
        out = out.dropna().reset_index(drop=True)
    else:
        feature_subset = [c for c in FEATURE_COLS if c in out.columns]
        out = out.dropna(subset=feature_subset).reset_index(drop=True)
    logger.info("Feature rows=%s cols=%s", len(out), len(out.columns))
    return out


def run_build_features() -> dict:
    src = data_dir() / "processed" / "peco_clean.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run preprocess first.")
    features = build_features(pd.read_csv(src))
    out_csv = data_dir() / "processed" / "peco_features.csv"
    out_parquet = data_dir() / "processed" / "peco_features.parquet"
    features.to_csv(out_csv, index=False)
    features.to_parquet(out_parquet, index=False)
    return {
        "csv": str(out_csv),
        "parquet": str(out_parquet),
        "rows": int(len(features)),
        "splits": features["split"].value_counts().to_dict(),
        "feature_cols": FEATURE_COLS,
    }
