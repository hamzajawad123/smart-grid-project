"""Append-only raw writes. Joined table goes to data/processed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smart_grid.config import data_dir


def new_ingest_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _raw_paths(kind: str, ingest_id: str) -> tuple[Path, Path]:
    raw = data_dir() / "raw"
    return (
        raw / f"{kind}_{ingest_id}.csv",
        raw / f"{kind}_{ingest_id}.parquet",
    )


def _manifest_path() -> Path:
    return data_dir() / "raw" / "manifest.json"


def save_raw(kind: str, frame: pd.DataFrame, ingest_id: str | None = None) -> dict:
    ingest_id = ingest_id or new_ingest_id()
    out = frame.copy()
    out["ingest_id"] = ingest_id
    csv_path, parquet_path = _raw_paths(kind, ingest_id)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() or parquet_path.exists():
        raise FileExistsError(f"Raw files already exist for {kind} {ingest_id}")
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)

    manifest = {}
    if _manifest_path().exists():
        manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    manifest[kind] = {
        "ingest_id": ingest_id,
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "rows": int(len(out)),
    }
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest[kind]


def load_latest(kind: str) -> pd.DataFrame:
    if not _manifest_path().exists():
        raise FileNotFoundError("No raw ingest yet. Run ingest_raw first.")
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    if kind not in manifest:
        raise FileNotFoundError(f"No raw {kind} ingest found.")
    parquet_path = Path(manifest[kind]["parquet"])
    csv_path = Path(manifest[kind]["csv"])
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing raw files for {kind}")


def load_joined() -> pd.DataFrame:
    energy = load_latest("energy")
    weather = load_latest("weather")
    weather_cols = [c for c in weather.columns if c not in {"ingest_id", "source"}]
    joined = energy.merge(
        weather[weather_cols],
        on=["grid_id", "ts_utc"],
        how="left",
        suffixes=("", "_weather"),
    )
    return joined.sort_values("ts_utc").reset_index(drop=True)


def processed_joined_path() -> Path:
    return data_dir() / "processed" / "peco_demand_weather_joined.csv"


def export_processed() -> dict:
    """Write stable files: raw energy/weather + processed joined CSV."""
    energy_path = data_dir() / "raw" / "peco_hourly_demand.csv"
    weather_path = data_dir() / "raw" / "philadelphia_hourly_weather.csv"
    energy = pd.read_csv(energy_path) if energy_path.exists() else load_latest("energy")
    weather = (
        pd.read_csv(weather_path) if weather_path.exists() else load_latest("weather")
    )
    energy["ts_utc"] = pd.to_datetime(energy["ts_utc"], utc=True)
    weather["ts_utc"] = pd.to_datetime(weather["ts_utc"], utc=True)
    weather_cols = [c for c in weather.columns if c not in {"ingest_id", "source"}]
    joined = energy.merge(
        weather[weather_cols],
        on=["grid_id", "ts_utc"],
        how="left",
        suffixes=("", "_weather"),
    ).sort_values("ts_utc")
    raw = data_dir() / "raw"
    energy_path = raw / "peco_hourly_demand.csv"
    weather_path = raw / "philadelphia_hourly_weather.csv"
    joined_path = processed_joined_path()
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    energy.to_csv(energy_path, index=False)
    weather.to_csv(weather_path, index=False)
    joined.to_csv(joined_path, index=False)
    return {
        "energy": str(energy_path),
        "weather": str(weather_path),
        "joined": str(joined_path),
    }


def load_joined_csv() -> pd.DataFrame:
    path = processed_joined_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ingest first.")
    frame = pd.read_csv(path)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame
