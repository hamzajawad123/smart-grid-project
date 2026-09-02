"""CLI: python -m scripts.ingest_raw  (or python scripts/ingest_raw.py)."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_grid.ingestion.pipeline import ingest_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = ingest_raw(incremental=not args.full)
    print(result)


if __name__ == "__main__":
    main()
