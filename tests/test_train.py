import pandas as pd
import pytest

from smart_grid.features.build import FEATURE_COLS, TARGET
from smart_grid.models.train import train_ensemble


def test_train_ensemble_rejects_empty_train_split():
    n = 8
    data = {col: [0.0] * n for col in FEATURE_COLS}
    data["ts_utc"] = pd.date_range("2026-09-01", periods=n, freq="h", tz="UTC")
    data[TARGET] = [4000.0] * n
    data["split"] = ["test"] * n
    with pytest.raises(ValueError, match="Train split is empty"):
        train_ensemble(pd.DataFrame(data))
