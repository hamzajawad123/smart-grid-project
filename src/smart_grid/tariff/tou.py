"""PECO Rate R Time-Of-Use generation charges. Separate from the demand model."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from smart_grid.config import load_params

PERIOD_LABELS = {
    "super_off_peak": "Super off-peak",
    "off_peak": "Off-peak",
    "peak": "Peak",
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_monday(year: int, month: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=cursor.weekday())


def is_peco_tou_holiday(local_day: date) -> bool:
    """Peak does not apply on PJM holidays (PECO Tariff No. 8 TOU option).

    Matches PECO's customer TOU list plus Memorial Day (NERC / PJM off-peak day):
    New Year's Day, MLK Day, Presidents Day, Memorial Day, Independence Day,
    Labor Day, Thanksgiving, day after Thanksgiving, Christmas Day.
    """
    year = local_day.year
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    holidays = {
        date(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_monday(year, 5),
        date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        thanksgiving,
        thanksgiving + timedelta(days=1),
        date(year, 12, 25),
    }
    return local_day in holidays


def period_for_ts(ts_utc) -> str:
    params = load_params()
    tariff = params["tariff"]
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(tariff.get("timezone", "America/New_York"))
    hour = int(local.hour)
    peak_hours = {int(h) for h in tariff["peak_hours_et"]}
    super_hours = {int(h) for h in tariff["super_off_peak_hours_et"]}
    if hour in super_hours:
        return "super_off_peak"
    weekday = local.dayofweek < 5
    if weekday and hour in peak_hours and not is_peco_tou_holiday(local.date()):
        return "peak"
    return "off_peak"


def rate_usd_per_kwh(period: str) -> float:
    tariff = load_params()["tariff"]
    mapping = {
        "peak": float(tariff["peak_usd"]),
        "off_peak": float(tariff["off_peak_usd"]),
        "super_off_peak": float(tariff["super_off_peak_usd"]),
    }
    return mapping.get(period, mapping["off_peak"])


def annotate_hours(hours: list[dict]) -> list[dict]:
    out = []
    for row in hours:
        period = period_for_ts(row["ts_utc"])
        item = dict(row)
        item["tariff_period"] = period
        item["rate_usd_per_kwh"] = rate_usd_per_kwh(period)
        out.append(item)
    return out


def recommend(hours: list[dict]) -> dict:
    """Point flexible load at PECO super off-peak (midnight–6am ET)."""
    hours = annotate_hours(hours)
    stressed = [h for h in hours if h.get("band") in {"high", "critical"}]
    cheapest = sorted(hours, key=lambda h: (h["rate_usd_per_kwh"], h.get("demand_mw", 0)))[:4]
    if stressed:
        action = "Shift flexible load out of weekday 2–6pm peak (and busy hours) into midnight–6am super off-peak."
    else:
        action = "No busy hours in this window. Prefer midnight–6am (PECO super off-peak) for large deferrable loads."
    tariff = load_params()["tariff"]
    return {
        "policy": "peco_rate_r_tou",
        "rate_class": tariff.get("rate_class", "R"),
        "component": "generation_supply",
        "effective": tariff.get("effective"),
        "through": tariff.get("through"),
        "note": (
            "PECO Rate R Time-Of-Use generation (GSA 1). Does not include distribution, "
            "transmission, or other bill riders. Hours are America/New_York."
        ),
        "action": action,
        "stress_hours_utc": [h["ts_utc"] for h in stressed],
        "suggested_off_peak_utc": [h["ts_utc"] for h in cheapest],
        "rates_usd_per_kwh": {
            "super_off_peak": rate_usd_per_kwh("super_off_peak"),
            "off_peak": rate_usd_per_kwh("off_peak"),
            "peak": rate_usd_per_kwh("peak"),
        },
    }
