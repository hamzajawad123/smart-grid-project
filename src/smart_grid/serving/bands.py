"""Capacity Stress Index bands. Statistical P75/P90/P95 — not physical overload."""

from __future__ import annotations


def stress_band(mw: float, p75: float, p90: float, p95: float) -> str:
    if mw >= p95:
        return "critical"
    if mw >= p90:
        return "high"
    if mw >= p75:
        return "elevated"
    return "normal"
