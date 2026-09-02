import pandas as pd

from smart_grid.features.build import build_features
from smart_grid.preprocessing.clean import preprocess


def _series() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=400, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_utc": idx,
            "grid_id": "PJM_PE",
            "demand_mw": range(400),
            "temperature_2m": 10.0,
            "relative_humidity_2m": 50.0,
            "wind_speed_10m": 5.0,
            "precipitation": 0.0,
            "cloud_cover": 20.0,
            "shortwave_radiation": 100.0,
            "source": "test",
            "ingest_id": "t",
        }
    )


def test_no_lag_1_column():
    cleaned = preprocess(_series())
    feats = build_features(cleaned)
    assert "lag_1" not in feats.columns
    assert "lag_24" in feats.columns
    assert "lag_168" in feats.columns


def test_lag_24_is_past_demand():
    cleaned = preprocess(_series())
    feats = build_features(cleaned)
    # After warmup drop, lag_24 equals demand 24 hours earlier.
    aligned = feats.set_index("ts_utc")
    y = cleaned.set_index("ts_utc")["demand_mw"]
    sample = aligned.index[10]
    assert aligned.loc[sample, "lag_24"] == y.loc[sample - pd.Timedelta(hours=24)]


def test_rolling_does_not_use_current_target():
    cleaned = preprocess(_series())
    feats = build_features(cleaned)
    row = feats.iloc[50]
    # roll_mean_24 is mean of demand from t-47..t-24, never t.
    ts = row["ts_utc"]
    y = cleaned.set_index("ts_utc")["demand_mw"]
    window = y.loc[ts - pd.Timedelta(hours=47) : ts - pd.Timedelta(hours=24)]
    assert abs(row["roll_mean_24"] - float(window.mean())) < 1e-6
