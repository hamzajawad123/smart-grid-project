"""EIA hourly demand for PJM sub-BA PE (PECO Energy zone)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd

from smart_grid.config import eia_api_key, load_params
from smart_grid.utils.http import get_json

logger = logging.getLogger(__name__)


def _parse_period(value: str) -> pd.Timestamp:
    # EIA hourly periods look like 2022-01-01T00 (UTC).
    return pd.to_datetime(value, utc=True)


def validate_subba(parent: str, subba: str) -> None:
    params = load_params()
    eia = params["eia"]
    url = f"{eia['base_url']}/{eia['route']}/facet/subba"
    payload = get_json(url, params={"api_key": eia_api_key(), "length": 5000})
    facets = payload.get("response", {}).get("facets", [])
    ids = {item.get("id") for item in facets}
    if not facets:
        logger.warning(
            "EIA facet list was empty; skipping subba validation for %s/%s",
            parent,
            subba,
        )
        return
    if subba not in ids:
        sample = sorted(str(i) for i in ids if i)[:30]
        raise ValueError(
            f"EIA subba={subba!r} was not found. "
            f"Expected EIA-930 code PE for PECO. Sample ids: {sample}"
        )
    logger.info("Validated EIA subba=%s (parent=%s)", subba, parent)


def fetch_demand(start: str | date, end: str | date | None = None) -> pd.DataFrame:
    params = load_params()
    eia = params["eia"]
    grid_id = params["grid_id"]
    parent = eia["parent"]
    subba = eia["subba"]
    validate_subba(parent, subba)

    end_str = (
        pd.Timestamp(end).strftime("%Y-%m-%dT%H")
        if end is not None
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    )
    start_str = pd.Timestamp(start).strftime("%Y-%m-%dT%H")
    if "T" not in str(start):
        start_str = pd.Timestamp(start).strftime("%Y-%m-%dT00")

    rows: list[dict] = []
    offset = 0
    page_size = int(eia["page_size"])
    while True:
        query = {
            "api_key": eia_api_key(),
            "frequency": eia["frequency"],
            "data[0]": "value",
            "facets[parent][]": parent,
            "facets[subba][]": subba,
            "start": start_str,
            "end": end_str,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size,
        }
        payload = get_json(f"{eia['base_url']}/{eia['route']}/data/", params=query)
        chunk = payload.get("response", {}).get("data", [])
        if not chunk:
            break
        rows.extend(chunk)
        logger.info("EIA demand rows so far: %s", len(rows))
        if len(chunk) < page_size:
            break
        offset += page_size

    if not rows:
        raise RuntimeError("EIA returned no demand rows for the requested window.")

    frame = pd.DataFrame(rows)
    frame["ts_utc"] = frame["period"].map(_parse_period)
    frame["demand_mw"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["grid_id"] = grid_id
    frame["source"] = "eia_region_sub_ba"
    out = (
        frame[["ts_utc", "grid_id", "demand_mw", "source"]]
        .dropna(subset=["ts_utc", "demand_mw"])
        .drop_duplicates(subset=["grid_id", "ts_utc"])
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )
    return out
