"""SHAP for the frozen mean ensemble: average TreeExplainer(LGBM) and TreeExplainer(XGB)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from smart_grid.config import models_dir
from smart_grid.features.build import FEATURE_COLS

logger = logging.getLogger(__name__)

GLOBAL_SAMPLE = 400


def _combine(lgbm, xgb, X: pd.DataFrame, weight_lgbm: float) -> tuple[np.ndarray, float]:
    frame = X[FEATURE_COLS]
    lgb_exp = shap.TreeExplainer(lgbm)
    xgb_exp = shap.TreeExplainer(xgb)
    lgb_s = np.asarray(lgb_exp.shap_values(frame), dtype=float)
    xgb_s = np.asarray(xgb_exp.shap_values(frame), dtype=float)
    shap_vals = weight_lgbm * lgb_s + (1.0 - weight_lgbm) * xgb_s
    base = float(
        weight_lgbm * np.asarray(lgb_exp.expected_value).reshape(-1)[0]
        + (1.0 - weight_lgbm) * np.asarray(xgb_exp.expected_value).reshape(-1)[0]
    )
    return shap_vals, base


def global_mean_abs(lgbm, xgb, X: pd.DataFrame, weight_lgbm: float = 0.5, max_rows: int = GLOBAL_SAMPLE) -> dict:
    if len(X) > max_rows:
        sample = X[FEATURE_COLS].sample(n=max_rows, random_state=42)
    else:
        sample = X[FEATURE_COLS]
    shap_vals, _ = _combine(lgbm, xgb, sample, weight_lgbm)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    ranked = {
        str(name): round(float(val), 4)
        for name, val in sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: -x[1])
    }
    logger.info("SHAP global sample=%s top=%s", len(sample), list(ranked)[:5])
    return {"n_rows": int(len(sample)), "mean_abs": ranked}


def local_rows(lgbm, xgb, X: pd.DataFrame, weight_lgbm: float = 0.5) -> list[dict]:
    shap_vals, base = _combine(lgbm, xgb, X, weight_lgbm)
    out = []
    for i in range(len(X)):
        contrib = {
            str(name): round(float(val), 3)
            for name, val in zip(FEATURE_COLS, shap_vals[i])
        }
        top = sorted(contrib.items(), key=lambda x: -abs(x[1]))[:8]
        out.append(
            {
                "base_mw": round(base, 2),
                "contributions": contrib,
                "top": [{"feature": k, "shap_mw": v} for k, v in top],
            }
        )
    return out


def write_global_to_pin(lgbm, xgb, X: pd.DataFrame, weight_lgbm: float, dest: Path | None = None) -> dict:
    dest = dest or models_dir()
    summary = global_mean_abs(lgbm, xgb, X, weight_lgbm=weight_lgbm)
    meta_path = dest / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["shap_global"] = summary
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (dest / "shap_global.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
