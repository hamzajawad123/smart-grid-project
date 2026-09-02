"""Load params.yaml and environment settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def project_root() -> Path:
    env = os.getenv("SMART_GRID_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in [here.parents[2], Path.cwd(), *Path.cwd().parents]:
        if (candidate / "params.yaml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find project root (params.yaml + src). "
        "Set SMART_GRID_ROOT or open the repo folder."
    )


def load_params() -> dict[str, Any]:
    path = project_root() / "params.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def eia_api_key() -> str:
    key = os.getenv("EIA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "EIA_API_KEY is missing. Get a free key at "
            "https://www.eia.gov/opendata/register.php and set it in .env or Colab."
        )
    return key


def data_dir() -> Path:
    root = project_root() / "data"
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    path = project_root() / "models" / "production"
    path.mkdir(parents=True, exist_ok=True)
    return path


def api_key() -> str:
    return os.getenv("API_KEY", "").strip()


def mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "").strip()
