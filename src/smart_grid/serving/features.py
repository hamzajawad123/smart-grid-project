"""24h-ahead inference features. Future demand is unknown; lags stay in the past."""

from __future__ import annotations

import pandas as pd

from smart_grid.features.build import FEATURE_COLS, TARGET, build_features
from smart_grid.preprocessing.clean import WEATHER_COLS


def horizon_features(
    history: pd.DataFrame,
    future_weather: pd.DataFrame,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    hist = history.copy()
    hist["ts_utc"] = pd.to_datetime(hist["ts_utc"], utc=True)
    hist = hist.sort_values("ts_utc")
    last_ts = hist["ts_utc"].max()
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz="UTC")

    weather = future_weather.copy()
    weather["ts_utc"] = pd.to_datetime(weather["ts_utc"], utc=True)
    weather = weather.drop_duplicates(subset=["ts_utc"]).set_index("ts_utc")

    future = pd.DataFrame({"ts_utc": future_index})
    future["grid_id"] = hist["grid_id"].iloc[-1]
    future[TARGET] = float("nan")
    future["was_imputed"] = 0
    future["split"] = "forecast"
    for col in WEATHER_COLS:
        if col in weather.columns:
            future[col] = weather.reindex(future_index)[col].to_numpy()
        elif col in hist.columns:
            future[col] = pd.NA

    hist_tail = hist.tail(400).copy()
    if "was_imputed" not in hist_tail.columns:
        hist_tail["was_imputed"] = 0
    combined = pd.concat([hist_tail, future], ignore_index=True)
    weather_cols = [c for c in WEATHER_COLS if c in combined.columns]
    combined[weather_cols] = combined[weather_cols].interpolate(limit=6).ffill().bfill()
    feats = build_features(combined, require_target=False)
    mask = feats["ts_utc"].isin(future_index)
    out = feats.loc[mask].copy()
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing inference features: {missing}")
    if len(out) != horizon_hours:
        raise ValueError(
            f"Expected {horizon_hours} forecast rows, got {len(out)}. "
            "Need more history or weather coverage."
        )
    return out
