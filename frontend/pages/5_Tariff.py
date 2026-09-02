from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get
from ui import area_line, band_bars, boot, callout, heading, kpis

boot(
    "Tariff",
    "When power is cheaper",
    "PECO Rate R Time-of-Use generation. Midnight–6am is cheapest. Weekdays 2–6pm cost the most.",
)

rec = get("/tariff-recommendation", params={"grid_id": "PJM_PE"})
fc = get("/forecast/PJM_PE")

if rec:
    rates = rec.get("rates_usd_per_kwh") or {}
    kpis(
        [
            {
                "label": "Super off-peak · 12–6am",
                "value": f"${rates.get('super_off_peak', 0):.4f}",
                "hint": "generation per kWh",
                "tone": "cyan",
            },
            {
                "label": "Off-peak · all other hours",
                "value": f"${rates.get('off_peak', 0):.4f}",
                "hint": "generation per kWh",
                "tone": "violet",
            },
            {
                "label": "Peak · weekdays 2–6pm",
                "value": f"${rates.get('peak', 0):.4f}",
                "hint": "generation per kWh",
                "tone": "pink",
            },
        ]
    )
    callout(rec.get("action") or "Prefer midnight–6am for large appliances.")

if not fc or not fc.get("hours"):
    callout("Open Next 24 hours and refresh a forecast to see tomorrow’s cheaper windows.")
    st.stop()

hours = pd.DataFrame(fc["hours"])
hours["ts_utc"] = pd.to_datetime(hours["ts_utc"])
heading("PECO generation price through the day", "Higher line = more expensive hour. Distribution is billed separately.")
area_line(hours, "ts_utc", "rate_usd_per_kwh", "Hour (UTC)", "Generation (USD / kWh)", color="#C4B5FD", y_format=".4f")

heading("Demand through the same day", "Orange and pink hours are the ones to avoid for big loads.")
band_bars(hours, "ts_utc", "demand_mw", "band", "Hour (UTC)", "Demand (MW)")
