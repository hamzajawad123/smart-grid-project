from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import get
from ui import boot, callout, friendly_feature, hbar, heading, kpis, pretty_when

boot(
    "Why",
    "What is driving demand",
    "See which conditions are pushing electricity use up or down for a chosen hour.",
)

exp = get("/explainability")
info = get("/model/info")
fc = get("/forecast/PJM_PE")

shap_global = None
if exp and exp.get("shap_global"):
    shap_global = exp["shap_global"]
elif info:
    shap_global = info.get("shap_global")

if shap_global and shap_global.get("mean_abs"):
    mean_abs = shap_global["mean_abs"]
    top_global = list(mean_abs.items())[:12]
    labels = [friendly_feature(k) for k, _ in top_global]
    values = [v for _, v in top_global]
    heading("What usually matters most", "Bigger bar = stronger typical effect on demand.")
    hbar(labels, values, "Typical effect (MW)", "Condition")

if not fc or not fc.get("hours"):
    callout("Open Next 24 hours and refresh a forecast to inspect a single hour.")
    st.stop()

hours = [h for h in fc["hours"] if h.get("shap")]
if not hours:
    callout("Refresh the 24h forecast to see what is driving each hour.")
    st.stop()

options = {f"{pretty_when(h['ts_utc'])} · {h['demand_mw']:.0f} MW": h for h in hours}
choice = st.selectbox("Choose an hour", list(options))
row = options[choice]
kpis(
    [
        {
            "label": "Expected demand",
            "value": f"{row['demand_mw']:.0f} MW",
            "hint": pretty_when(row["ts_utc"]),
            "tone": "lime",
        },
        {
            "label": "Time-of-use window",
            "value": {
                "super_off_peak": "Super off-peak",
                "off_peak": "Off-peak",
                "peak": "Peak",
            }.get(row.get("tariff_period"), "—"),
            "hint": "PECO Rate R period",
            "tone": "violet",
        },
    ]
)

top = row["shap"]["top"]
heading("This hour", "Green raises demand. Pink lowers it.")
hbar(
    [friendly_feature(item["feature"]) for item in top],
    [float(item["shap_mw"]) for item in top],
    "Effect on demand (MW)",
    "Condition",
    signed=True,
)
