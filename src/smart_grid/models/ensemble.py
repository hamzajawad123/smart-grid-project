"""Locked production model: mean of LightGBM and XGBoost."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MeanEnsemble:
    """0.5 * LightGBM + 0.5 * XGBoost (weight configurable)."""

    def __init__(self, lgbm: Any, xgb: Any, weight_lgbm: float = 0.5):
        self.lgbm = lgbm
        self.xgb = xgb
        self.weight_lgbm = float(weight_lgbm)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p_lgb = np.asarray(self.lgbm.predict(X), dtype=float)
        p_xgb = np.asarray(self.xgb.predict(X), dtype=float)
        return self.weight_lgbm * p_lgb + (1.0 - self.weight_lgbm) * p_xgb
