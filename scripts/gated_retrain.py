"""Refit locked ensemble. Promote to models/production/ only if test WAPE improves."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_grid.config import load_params, models_dir, project_root  # noqa: E402
from smart_grid.models.promotion import should_promote  # noqa: E402
from smart_grid.models.registry import register_ensemble  # noqa: E402
from smart_grid.models.train import save_production_pin, train_ensemble  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Promote even if worse")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    params = load_params()
    wape_tie = float(params.get("promotion", {}).get("wape_tie", 0.002))
    prod_dir = models_dir()
    prod_meta_path = prod_dir / "metadata.json"
    if not prod_meta_path.exists():
        raise FileNotFoundError(f"No production pin at {prod_dir}")
    production = json.loads(prod_meta_path.read_text(encoding="utf-8"))

    bundle = train_ensemble()
    candidate_dir = project_root() / "models" / "candidate"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    save_production_pin(bundle, dest=candidate_dir)
    candidate = bundle["metadata"]
    promote, reason = should_promote(candidate, production, wape_tie)
    if args.force:
        promote, reason = True, "forced"

    decision = {
        "promoted": promote,
        "reason": reason,
        "candidate_test": candidate["refit_metrics"]["test"],
        "production_test": production["refit_metrics"]["test"],
    }
    (candidate_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))

    if not promote:
        logging.info("Not promoting: %s", reason)
        return

    save_production_pin(bundle, dest=prod_dir)
    try:
        info = register_ensemble(bundle)
        decision["mlflow"] = info
        print("mlflow", info)
    except Exception as exc:
        logging.exception("MLflow register failed; local pin was still updated: %s", exc)


if __name__ == "__main__":
    main()
