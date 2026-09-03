"""Shared FastAPI client for Streamlit. On Community Cloud, calls serving code in-process."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SMART_GRID_ROOT", str(ROOT))
src = str(ROOT / "src")
if src not in sys.path:
    sys.path.insert(0, src)

_SECRET_KEYS = (
    "EIA_API_KEY",
    "SMART_GRID_API_URL",
    "SMART_GRID_ROOT",
    "API_KEY",
    "SMART_GRID_SERVE",
)


def _apply_streamlit_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in _SECRET_KEYS:
        try:
            value = secrets[key]
        except Exception:
            continue
        if value is not None and not os.getenv(key):
            os.environ[key] = str(value)


_apply_streamlit_secrets()

API_URL = os.getenv("SMART_GRID_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "").strip()


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


def _use_http() -> bool:
    """Docker / a public API uses HTTP. Streamlit Cloud and local UI-only stay in-process."""
    mode = os.getenv("SMART_GRID_SERVE", "").strip().lower()
    if mode in {"local", "inprocess", "streamlit"}:
        return False
    if mode in {"api", "http"}:
        return True
    host = (urlparse(API_URL).hostname or "").lower()
    if host in {"", "127.0.0.1", "localhost"}:
        return False
    return True


def _local(method: str, path: str, params: dict | None = None, body: dict | None = None):
    from smart_grid.serving.dispatch import ServingError, call

    try:
        with st.spinner("Loading PECO demand data…"):
            return call(
                method,
                path,
                params=params,
                body=body,
                bootstrap=True,
                create_forecast_if_missing=True,
            )
    except ServingError as exc:
        if exc.status == 404:
            return None
        st.error(exc.message[:400])
        return None
    except Exception as exc:
        st.error(str(exc)[:400])
        return None


def get(path: str, params: dict | None = None):
    if not _use_http():
        return _local("GET", path, params=params)
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, headers=_headers(), timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.error(f"API {exc.response.status_code}: {detail[:400]}")
        return None
    except httpx.RequestError as exc:
        st.error(f"Cannot reach API at {API_URL}. Start it with `python scripts/run_api.py`. ({exc})")
        return None


def post(path: str, json: dict):
    if not _use_http():
        return _local("POST", path, body=json)
    try:
        response = httpx.post(f"{API_URL}{path}", json=json, headers=_headers(), timeout=120.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API {exc.response.status_code}: {exc.response.text[:400]}")
        return None
    except httpx.RequestError as exc:
        st.error(f"Cannot reach API at {API_URL}. Start it with `python scripts/run_api.py`. ({exc})")
        return None
