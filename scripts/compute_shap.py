"""CLI: python scripts/compute_shap.py — global SHAP on the frozen pin (no retrain)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_grid.features.build import FEATURE_COLS  # noqa: E402
from smart_grid.models.explain import write_global_to_pin  # noqa: E402
from smart_grid.models.train import _load_features, load_production_pin  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pin = load_production_pin()
    if pin.get("lgbm") is None or pin.get("xgb") is None:
        raise FileNotFoundError("Need models/production/lgbm.joblib and xgb.joblib")
    train = _load_features()
    train = train[train["split"] == "train"]
    weight = float(pin["metadata"].get("ensemble_weight_lgbm", 0.5))
    summary = write_global_to_pin(
        pin["lgbm"],
        pin["xgb"],
        train[FEATURE_COLS],
        weight_lgbm=weight,
    )
    top = list(summary["mean_abs"].items())[:8]
    print({"n_rows": summary["n_rows"], "top": top})


if __name__ == "__main__":
    main()
