import numpy as np
import pandas as pd

from smart_grid.models.ensemble import MeanEnsemble
from smart_grid.preprocessing.clean import WEATHER_COLS
from smart_grid.serving.features import horizon_features


class _Const:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_mean_ensemble_averages():
    model = MeanEnsemble(_Const(100), _Const(50), weight_lgbm=0.5)
    pred = model.predict(np.zeros((3, 2)))
    assert np.allclose(pred, 75)


def _history(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts_utc": idx,
            "grid_id": "PJM_PE",
            "demand_mw": np.arange(n, dtype=float),
            "was_imputed": 0,
        }
    )
    for col in WEATHER_COLS:
        frame[col] = 10.0
    return frame


def test_horizon_lags_do_not_use_future_demand():
    hist = _history()
    last = hist["ts_utc"].max()
    future_idx = pd.date_range(last + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame({"ts_utc": future_idx})
    for col in WEATHER_COLS:
        weather[col] = 12.0
    feats = horizon_features(hist, weather, horizon_hours=24)
    y = hist.set_index("ts_utc")["demand_mw"]
    first = feats.iloc[0]
    ts = first["ts_utc"]
    assert first["lag_24"] == y.loc[ts - pd.Timedelta(hours=24)]
    assert first["demand_mw"] != first["demand_mw"]  # NaN target
    last_row = feats.iloc[-1]
    ts_last = last_row["ts_utc"]
    assert last_row["lag_24"] == y.loc[ts_last - pd.Timedelta(hours=24)]
