from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get, post
from ui import area_line, boot, callout, heading, scatter

boot(
    "Weather",
    "Weather and demand",
    "Warmer hours usually mean more electricity. See temperature next to demand.",
)

payload = get("/forecast/PJM_PE")
if not payload:
    payload = post("/forecast", {"grid_id": "PJM_PE", "horizon_hours": 24, "include_shap": True})
hist = get("/historical/PJM_PE", params={"limit": 168})

if payload and payload.get("hours"):
    hours = pd.DataFrame(payload["hours"])
    hours["ts_utc"] = pd.to_datetime(hours["ts_utc"])
    heading("Temperature vs demand, next 24 hours", "Each dot is one hour.")
    scatter(hours, "temperature_2m", "demand_mw", "Temperature (°C)", "Demand (MW)", color_col="demand_mw")
    heading("Demand, next 24 hours", "")
    area_line(hours, "ts_utc", "demand_mw", "Hour (UTC)", "Demand (MW)")
    heading("Temperature, next 24 hours", "Philadelphia.")
    area_line(hours, "ts_utc", "temperature_2m", "Hour (UTC)", "Temperature (°C)", color="#5CE1FF", y_format=".1f")
else:
    callout("Open Next 24 hours and refresh a forecast first.")

if hist and hist.get("rows"):
    frame = pd.DataFrame(hist["rows"])
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"])
    heading("Demand, last 7 days", "What was actually used.")
    area_line(frame, "ts_utc", "demand_mw", "Time (UTC)", "Demand (MW)")
    if "temperature_2m" in frame.columns:
        heading("Temperature, last 7 days", "Philadelphia.")
        area_line(
            frame,
            "ts_utc",
            "temperature_2m",
            "Time (UTC)",
            "Temperature (°C)",
            color="#5CE1FF",
            y_format=".1f",
        )
