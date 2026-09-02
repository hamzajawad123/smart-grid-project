import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from smart_grid.features.build import FEATURE_COLS
from smart_grid.models.ensemble import MeanEnsemble
from smart_grid.models.explain import global_mean_abs, local_rows


def _toy_models():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = 4000 + 2 * X["lag_24"] + 3 * X["temperature_2m"]
    lgbm = LGBMRegressor(n_estimators=10, num_leaves=8, verbose=-1, random_state=0)
    xgb = XGBRegressor(n_estimators=10, max_depth=3, verbosity=0, random_state=0, tree_method="hist")
    lgbm.fit(X, y)
    xgb.fit(X, y)
    return lgbm, xgb, X


def test_global_shap_ranks_features():
    lgbm, xgb, X = _toy_models()
    summary = global_mean_abs(lgbm, xgb, X, max_rows=40)
    assert summary["n_rows"] == 40
    assert "lag_24" in summary["mean_abs"]
    assert summary["mean_abs"]["lag_24"] >= 0


def test_local_shap_adds_toward_prediction():
    lgbm, xgb, X = _toy_models()
    row = X.head(1)
    local = local_rows(lgbm, xgb, row)[0]
    reconstructed = local["base_mw"] + sum(local["contributions"].values())
    pred = float(MeanEnsemble(lgbm, xgb).predict(row)[0])
    assert abs(reconstructed - pred) < 5
    assert local["top"]
