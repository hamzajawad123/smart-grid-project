"""Refresh EIA + weather, then rebuild clean/feature tables."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_grid.features.build import run_build_features  # noqa: E402
from smart_grid.ingestion.pipeline import ingest_raw  # noqa: E402
from smart_grid.preprocessing.clean import run_preprocess  # noqa: E402
from smart_grid.serving.monitoring import drift_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Re-fetch from params start_date")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest = ingest_raw(incremental=not args.full)
    prep = run_preprocess()
    feats = run_build_features()
    drift = drift_summary()
    print(
        {
            "ingest": {
                "last_demand_utc": ingest.get("last_demand_utc"),
                "energy_rows": ingest.get("energy_rows"),
            },
            "preprocess": prep,
            "features": {"rows": feats["rows"], "splits": feats["splits"]},
            "drift_any_flag": drift.get("any_flag"),
        }
    )


if __name__ == "__main__":
    main()
