from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get
from ui import area_line, boot, heading, kpis, pretty_when

boot(
    "PECO Demand",
    "PECO demand",
    "See how much electricity the Philadelphia area is using, and what the next day looks like.",
)

hist = get("/historical/PJM_PE", params={"limit": 168})
fc = get("/forecast/PJM_PE")

if hist and hist.get("rows"):
    frame = pd.DataFrame(hist["rows"])
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"])
    latest = frame.iloc[-1]
    cards = [
        {
            "label": "Latest demand",
            "value": f"{latest['demand_mw']:.0f} MW",
            "hint": pretty_when(latest["ts_utc"]),
            "tone": "lime",
        },
        {
            "label": "Week’s highest hour",
            "value": f"{frame['demand_mw'].max():.0f} MW",
            "hint": pretty_when(frame.loc[frame["demand_mw"].idxmax(), "ts_utc"]),
            "tone": "pink",
        },
        {
            "label": "Week’s lowest hour",
            "value": f"{frame['demand_mw'].min():.0f} MW",
            "hint": pretty_when(frame.loc[frame["demand_mw"].idxmin(), "ts_utc"]),
            "tone": "cyan",
        },
    ]
    if "temperature_2m" in frame.columns and pd.notna(latest.get("temperature_2m")):
        cards.append(
            {
                "label": "Latest temperature",
                "value": f"{float(latest['temperature_2m']):.0f} °C",
                "hint": "Philadelphia",
                "tone": "violet",
            }
        )
    kpis(cards)

    heading("Demand over the last 7 days", "Actual electricity use, hour by hour.")
    area_line(frame, "ts_utc", "demand_mw", "Time (UTC)", "Demand (MW)")
    if "temperature_2m" in frame.columns:
        heading("Temperature over the last 7 days", "Philadelphia weather for the same week.")
        area_line(
            frame,
            "ts_utc",
            "temperature_2m",
            "Time (UTC)",
            "Temperature (°C)",
            color="#5CE1FF",
            y_format=".1f",
        )

if fc and fc.get("hours"):
    hours = pd.DataFrame(fc["hours"])
    hours["ts_utc"] = pd.to_datetime(hours["ts_utc"])
    peak_row = hours.loc[hours["demand_mw"].idxmax()]
    peak_when = pretty_when(peak_row["ts_utc"])
    heading("Coming up", "Open Next 24 hours for the full forecast.")
    kpis(
        [
            {
                "label": "Highest hour tomorrow",
                "value": f"{peak_row['demand_mw']:.0f} MW",
                "hint": peak_when,
                "tone": "orange",
            }
        ]
    )
