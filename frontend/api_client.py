"""Shared FastAPI client for Streamlit."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("SMART_GRID_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "").strip()


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


def get(path: str, params: dict | None = None):
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
