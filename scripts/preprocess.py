"""CLI: python scripts/preprocess.py"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_grid.preprocessing.clean import run_preprocess  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run_preprocess())


if __name__ == "__main__":
    main()
