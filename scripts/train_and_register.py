"""Fit locked ensemble, write models/production/, log to MLflow (DagsHub if URI set)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_grid.features.build import FEATURE_COLS  # noqa: E402
from smart_grid.models.registry import register_ensemble  # noqa: E402
from smart_grid.models.train import save_production_pin, train_ensemble  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = train_ensemble()
    save_production_pin(bundle)
    sample = pd.read_csv(
        ROOT / "data" / "processed" / "peco_features.csv",
        nrows=40,
    )[FEATURE_COLS]
    try:
        info = register_ensemble(bundle, sample_x=sample)
        print("mlflow", info)
    except Exception as exc:
        logging.exception("MLflow register failed; local pin is still valid: %s", exc)
        info = {"error": str(exc)}
    print("pin", ROOT / "models" / "production")
    print("refit test", bundle["metadata"]["refit_metrics"]["test"])
    print("colab test (selection)", bundle["metadata"]["colab_selection_metrics"]["ensemble_mean"])


if __name__ == "__main__":
    main()
