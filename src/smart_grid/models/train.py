"""Fit the frozen LightGBM + XGBoost ensemble (locked Colab hyperparameters)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from smart_grid.config import data_dir, load_params, models_dir
from smart_grid.features.build import FEATURE_COLS, TARGET
from smart_grid.models.ensemble import MeanEnsemble
from smart_grid.models.explain import global_mean_abs
from smart_grid.models.metrics import score_frame
from smart_grid.models.selection import COLAB_TEST_METRICS, WINNER

logger = logging.getLogger(__name__)


def _load_features() -> pd.DataFrame:
    csv_path = data_dir() / "processed" / "peco_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run scripts/build_features.py first.")
    frame = pd.read_csv(csv_path)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame.sort_values("ts_utc").reset_index(drop=True)


def _lgbm(params: dict) -> LGBMRegressor:
    cfg = params["model"]["lightgbm"]
    return LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        num_leaves=int(cfg["num_leaves"]),
        min_child_samples=int(cfg["min_child_samples"]),
        subsample=float(cfg["subsample"]),
    )


def _xgb(params: dict) -> XGBRegressor:
    cfg = params["model"]["xgboost"]
    return XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        max_depth=int(cfg["max_depth"]),
        min_child_weight=float(cfg["min_child_weight"]),
        subsample=float(cfg["subsample"]),
    )


def train_ensemble(frame: pd.DataFrame | None = None) -> dict:
    params = load_params()
    df = frame if frame is not None else _load_features()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    x_train, y_train = train[FEATURE_COLS], train[TARGET]
    p90 = float(y_train.quantile(0.90))
    p75 = float(y_train.quantile(0.75))
    p95 = float(y_train.quantile(0.95))

    lgbm = _lgbm(params)
    xgb = _xgb(params)
    logger.info("Fitting LightGBM on %s train rows", len(train))
    lgbm.fit(x_train, y_train)
    logger.info("Fitting XGBoost on %s train rows", len(train))
    xgb.fit(x_train, y_train)

    weight = float(params["model"].get("ensemble_weight_lgbm", 0.5))
    ensemble = MeanEnsemble(lgbm, xgb, weight_lgbm=weight)

    scores = {}
    for split_name, part in [("val", val), ("test", test)]:
        pred = ensemble.predict(part[FEATURE_COLS])
        scores[split_name] = score_frame(part[TARGET], pred, p90)

    importance = {}
    names = getattr(lgbm, "feature_name_", FEATURE_COLS)
    gains = getattr(lgbm, "feature_importances_", None)
    if gains is not None:
        importance = {
            str(name): float(gain)
            for name, gain in sorted(zip(names, gains), key=lambda x: -x[1])
        }

    metadata = {
        "winner": WINNER,
        "grid_id": params["grid_id"],
        "horizon_hours": int(params.get("horizon_hours", 24)),
        "feature_cols": FEATURE_COLS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "thresholds_mw": {"p75": p75, "p90": p90, "p95": p95},
        "refit_metrics": scores,
        "colab_selection_metrics": COLAB_TEST_METRICS,
        "lightgbm_params": params["model"]["lightgbm"],
        "xgboost_params": params["model"]["xgboost"],
        "ensemble_weight_lgbm": weight,
        "feature_importance_lgbm": importance,
        "shap_global": None,
    }
    try:
        metadata["shap_global"] = global_mean_abs(lgbm, xgb, x_train, weight_lgbm=weight)
    except Exception:
        logger.exception("SHAP global failed; pin still valid without it")
    return {"ensemble": ensemble, "lgbm": lgbm, "xgb": xgb, "metadata": metadata}


def save_production_pin(bundle: dict, dest: Path | None = None) -> Path:
    dest = dest or models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["ensemble"], dest / "ensemble.joblib")
    joblib.dump(bundle["lgbm"], dest / "lgbm.joblib")
    joblib.dump(bundle["xgb"], dest / "xgb.joblib")
    (dest / "metadata.json").write_text(
        json.dumps(bundle["metadata"], indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote production pin to %s", dest)
    return dest


def load_production_pin(dest: Path | None = None) -> dict:
    dest = dest or models_dir()
    ensemble_path = dest / "ensemble.joblib"
    meta_path = dest / "metadata.json"
    if not ensemble_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No production model at {dest}. Run python scripts/train_and_register.py"
        )
    return {
        "ensemble": joblib.load(ensemble_path),
        "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
        "lgbm": joblib.load(dest / "lgbm.joblib") if (dest / "lgbm.joblib").exists() else None,
        "xgb": joblib.load(dest / "xgb.joblib") if (dest / "xgb.joblib").exists() else None,
    }
