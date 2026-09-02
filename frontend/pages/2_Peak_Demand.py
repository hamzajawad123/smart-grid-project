from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get
from ui import band_bars, boot, callout, chips, heading, kpis, pretty_when

boot(
    "Peak Demand",
    "Busy hours",
    "See when demand is highest in the next 24 hours, and which hours to watch.",
)

fc = get("/forecast/PJM_PE")
alerts = get("/peak-alerts")

if not fc or not fc.get("hours"):
    callout("Open Next 24 hours and refresh a forecast first.")
    st.stop()

hours = pd.DataFrame(fc["hours"])
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

heading("Next 24 hours", "Cyan is quiet. Lime is picking up. Orange and pink are the peak hours.")
band_bars(hours, "ts_utc", "demand_mw", "band", "Hour (UTC)", "Demand (MW)")

rows = (alerts or {}).get("alerts") or []
if not rows:
    callout("No extra-busy hours in this forecast.")
else:
    heading("Hours to watch", "These are the peak windows.")
    chips(
        [
            {
                "label": pretty_when(r.get("ts_utc", "")),
                "value": f"{r.get('demand_mw', 0):.0f} MW",
            }
            for r in rows
        ]
    )
