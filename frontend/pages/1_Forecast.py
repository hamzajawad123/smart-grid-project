from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get, post
from ui import area_line, band_bars, boot, callout, heading, kpis, pretty_when

boot(
    "Forecast",
    "Next 24 hours",
    "Expected electricity use for the coming day. Refresh to update.",
)

go = st.button("Refresh 24h forecast", type="primary")
payload = (
    post("/forecast", {"grid_id": "PJM_PE", "horizon_hours": 24, "include_shap": True})
    if go
    else get("/forecast/PJM_PE")
)

if not payload:
    callout("Press the red button to load the next 24 hours.")
    st.stop()

hours = pd.DataFrame(payload["hours"])
hours["ts_utc"] = pd.to_datetime(hours["ts_utc"])
peak = hours.loc[hours["demand_mw"].idxmax()]
busy = hours[hours["band"].isin(["high", "critical"])]
kpis(
    [
        {
            "label": "Highest hour",
            "value": f"{peak['demand_mw']:.0f} MW",
            "hint": pretty_when(peak["ts_utc"]),
            "tone": "pink",
        },
        {
            "label": "Lowest hour",
            "value": f"{hours['demand_mw'].min():.0f} MW",
            "hint": pretty_when(hours.loc[hours["demand_mw"].idxmin(), "ts_utc"]),
            "tone": "cyan",
        },
        {
            "label": "Busy hours",
            "value": str(len(busy)),
            "hint": (
                " · ".join(pretty_when(ts) for ts in busy["ts_utc"])
                if len(busy)
                else "none in the next 24 hours"
            ),
            "tone": "orange",
        },
    ]
)

heading("Demand by hour", "Taller bars mean more electricity. Orange and pink are the busiest hours.")
band_bars(hours, "ts_utc", "demand_mw", "band", "Hour (UTC)", "Demand (MW)")

heading("Demand through the day", "The same 24 hours as a smooth curve.")
area_line(hours, "ts_utc", "demand_mw", "Hour (UTC)", "Demand (MW)")

if "temperature_2m" in hours.columns:
    heading("Temperature through the day", "How warm it is expected to be in Philadelphia.")
    area_line(hours, "ts_utc", "temperature_2m", "Hour (UTC)", "Temperature (°C)", color="#5CE1FF", y_format=".1f")
