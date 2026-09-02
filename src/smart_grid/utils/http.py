"""Tiny HTTP helper with retries."""

from __future__ import annotations

import time
from typing import Any

import requests


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 4,
    timeout: int = 90,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2 ** attempt)
                last_error = RuntimeError(
                    f"HTTP {response.status_code} for {url}: {response.text[:300]}"
                )
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {retries} tries: {url}") from last_error
