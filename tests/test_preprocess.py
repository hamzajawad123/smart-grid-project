import pandas as pd

from smart_grid.preprocessing.clean import preprocess


def _toy() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=200, freq="h", tz="UTC")
    demand = pd.Series(range(200), dtype=float)
    demand.iloc[10:12] = None
    return pd.DataFrame(
        {
            "ts_utc": idx,
            "grid_id": "PJM_PE",
            "demand_mw": demand.values,
            "temperature_2m": 10.0,
            "source": "test",
            "ingest_id": "t",
        }
    )


def test_fills_gaps_and_keeps_hourly():
    cleaned = preprocess(_toy())
    assert cleaned["demand_mw"].isna().sum() == 0
    full = pd.date_range(cleaned["ts_utc"].min(), cleaned["ts_utc"].max(), freq="h", tz="UTC")
    assert len(cleaned) == len(full)
    assert int(cleaned["was_imputed"].sum()) == 2


def test_does_not_clip_peaks():
    frame = _toy()
    frame.loc[5, "demand_mw"] = 8653.0
    cleaned = preprocess(frame)
    assert cleaned["demand_mw"].max() == 8653.0


def test_split_is_chronological():
    cleaned = preprocess(_toy())
    order = {"train": 0, "val": 1, "test": 2}
    codes = cleaned["split"].map(order)
    assert codes.is_monotonic_increasing or cleaned["split"].nunique() == 1
