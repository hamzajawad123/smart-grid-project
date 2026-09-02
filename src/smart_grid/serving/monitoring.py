"""Input drift vs train. PSI + z-scores. Optional Evidently if installed."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from smart_grid.config import data_dir, models_dir
from smart_grid.models.train import _load_features

WATCH = ["demand_mw", "temperature_2m", "lag_24", "lag_168", "hdd", "cdd"]


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return float("nan")
    lo = float(min(expected.min(), actual.min()))
    hi = float(max(expected.max(), actual.max()))
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    e_pct = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a_pct = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_summary() -> dict:
    feats = _load_features()
    train = feats[feats["split"] == "train"]
    recent = feats.tail(168)
    out: dict = {
        "window_hours": 168,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": {},
        "engine": "builtin_psi",
    }
    for col in WATCH:
        if col not in feats.columns:
            continue
        train_mean = float(train[col].mean())
        recent_mean = float(recent[col].mean())
        train_std = float(train[col].std()) or 1.0
        z = (recent_mean - train_mean) / train_std
        psi = _psi(train[col], recent[col])
        out["features"][col] = {
            "train_mean": train_mean,
            "recent_mean": recent_mean,
            "z_vs_train": round(z, 3),
            "psi": round(psi, 4) if psi == psi else None,
            "flag": bool(abs(z) > 2 or (psi == psi and psi > 0.2)),
        }
    out["any_flag"] = any(v["flag"] for v in out["features"].values())

    pin_meta = models_dir() / "metadata.json"
    if pin_meta.exists():
        meta = json.loads(pin_meta.read_text(encoding="utf-8"))
        shap = meta.get("shap_global") or {}
        mean_abs = shap.get("mean_abs") or {}
        out["shap_top"] = dict(list(mean_abs.items())[:8])

    try:
        import evidently  # noqa: F401

        out["engine"] = "builtin_psi+evidently_installed"
    except Exception:
        pass

    report_path = data_dir() / "processed" / "drift_report.json"
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["report_path"] = str(report_path)
    return out
