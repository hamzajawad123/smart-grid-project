"""Shared Streamlit theme, KPI cards, and Plotly charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LIME = "#C8FF3D"
CYAN = "#5CE1FF"
VIOLET = "#C4B5FD"
PINK = "#FF6B9A"
ORANGE = "#FFB020"
INK = "#F4F4F5"
MUTED = "#D4D4D8"
GRID = "rgba(255,255,255,0.08)"
PLOT_BG = "rgba(18, 18, 24, 0.0)"
PAPER = "rgba(0,0,0,0)"
FONT = "Times New Roman, Times, serif"

BAND_COLOR = {
    "normal": CYAN,
    "elevated": LIME,
    "high": ORANGE,
    "critical": PINK,
}

FEATURE_LABELS = {
    "lag_24": "Demand yesterday (same hour)",
    "lag_48": "Demand 2 days ago",
    "lag_72": "Demand 3 days ago",
    "lag_168": "Demand last week",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "wind_speed_10m": "Wind speed",
    "precipitation": "Precipitation",
    "cloud_cover": "Cloud cover",
    "shortwave_radiation": "Solar radiation",
    "hdd": "Heating need",
    "cdd": "Cooling need",
    "temp_change_24": "24h temperature change",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day (cycle)",
    "dow_sin": "Day of week",
    "dow_cos": "Day of week (cycle)",
    "month_sin": "Month",
    "month_cos": "Month (cycle)",
    "is_weekend": "Weekend",
    "is_holiday": "Holiday",
    "roll_mean_24": "Typical last-day level",
    "roll_std_24": "How jumpy yesterday was",
    "roll_min_24": "Quietest recent hour",
    "roll_max_24": "Busiest recent hour",
    "roll_mean_168": "Typical last-week level",
    "was_imputed": "Filled gap hour",
}

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stText,
.stButton, .stSelectbox, label, p, h1, h2, h3 {
  font-family: "Times New Roman", Times, serif !important;
}
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(200,255,61,0.12), transparent 50%),
    radial-gradient(900px 500px at 100% 0%, rgba(92,225,255,0.10), transparent 45%),
    radial-gradient(800px 400px at 80% 100%, rgba(255,107,154,0.10), transparent 40%),
    #07070b;
  color: #F4F4F5;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101018 0%, #07070b 100%);
  border-right: 1px solid rgba(200,255,61,0.12);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

div.stButton > button,
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: #E11D48 !important;
  color: #FFFFFF !important;
  border: 1px solid #FB7185 !important;
  font-family: "Times New Roman", Times, serif !important;
  font-size: 1.08rem !important;
  font-weight: 700 !important;
  padding: 0.55rem 1.2rem !important;
  border-radius: 12px !important;
}
div.stButton > button:hover {
  background: #BE123C !important;
  color: #FFFFFF !important;
}

.hero-banner {
  border-radius: 20px;
  padding: 22px 26px 18px;
  margin: 0 0 1.5rem 0;
  background:
    linear-gradient(135deg, rgba(200,255,61,0.22) 0%, rgba(92,225,255,0.12) 55%, rgba(196,181,253,0.10) 100%);
  border: 1px solid rgba(200,255,61,0.38);
  box-shadow: 0 12px 32px rgba(0,0,0,0.35);
}
.hero {
  font-family: "Times New Roman", Times, serif;
  font-weight: 700;
  font-size: 2.5rem;
  margin: 0;
  color: #C8FF3D;
  line-height: 1.15;
}
.blurb {
  color: #F4F4F5;
  font-size: 1.15rem;
  margin: 0.4rem 0 0 0;
}
.chart-head {
  font-family: "Times New Roman", Times, serif;
  font-size: 1.45rem;
  font-weight: 700;
  margin: 1.6rem 0 0.35rem 0;
  color: #FAFAFA;
}
.chart-sub { color: #D4D4D8; font-size: 1.05rem; margin: 0 0 0.6rem 0; }

.kpi-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 0.4rem 0 1.2rem 0;
}
.kpi {
  flex: 1 1 180px;
  min-width: 180px;
  min-height: 118px;
  border-radius: 18px;
  padding: 16px 18px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 28px rgba(0,0,0,0.35);
}
.kpi.lime { background: linear-gradient(160deg, rgba(200,255,61,0.22), rgba(255,255,255,0.03)); border-color: rgba(200,255,61,0.35); }
.kpi.cyan { background: linear-gradient(160deg, rgba(92,225,255,0.20), rgba(255,255,255,0.03)); border-color: rgba(92,225,255,0.35); }
.kpi.violet { background: linear-gradient(160deg, rgba(196,181,253,0.22), rgba(255,255,255,0.03)); border-color: rgba(196,181,253,0.35); }
.kpi.pink { background: linear-gradient(160deg, rgba(255,107,154,0.22), rgba(255,255,255,0.03)); border-color: rgba(255,107,154,0.35); }
.kpi.orange { background: linear-gradient(160deg, rgba(255,176,32,0.22), rgba(255,255,255,0.03)); border-color: rgba(255,176,32,0.35); }
.kpi-label { font-size: 0.95rem; color: #E4E4E7; font-weight: 700; }
.kpi-value { font-family: "Times New Roman", Times, serif; font-size: 1.85rem; font-weight: 700; color: #FAFAFA; margin-top: 4px; }
.kpi-hint { font-size: 0.9rem; color: #D4D4D8; margin-top: 4px; line-height: 1.35; }

.callout {
  border-radius: 18px;
  padding: 16px 18px;
  margin: 0.6rem 0 1rem 0;
  border: 1px solid rgba(200,255,61,0.28);
  background: linear-gradient(90deg, rgba(200,255,61,0.12), rgba(92,225,255,0.08));
  color: #F4F4F5;
  font-size: 1.1rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 0.5rem 0 1rem 0; }
.chip {
  flex: 1 1 180px;
  min-width: 180px;
  min-height: 118px;
  border-radius: 18px;
  padding: 16px 18px;
  background: linear-gradient(160deg, rgba(255,107,154,0.22), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,107,154,0.35);
  box-shadow: 0 10px 28px rgba(0,0,0,0.35);
}
.chip:nth-child(3n+2) {
  background: linear-gradient(160deg, rgba(255,176,32,0.22), rgba(255,255,255,0.03));
  border-color: rgba(255,176,32,0.35);
}
.chip:nth-child(3n) {
  background: linear-gradient(160deg, rgba(200,255,61,0.22), rgba(255,255,255,0.03));
  border-color: rgba(200,255,61,0.35);
}
.chip .t { font-size: 0.95rem; color: #E4E4E7; font-weight: 700; }
.chip .v { font-family: "Times New Roman", Times, serif; font-weight: 700; font-size: 1.85rem; color: #FAFAFA; margin-top: 4px; }
</style>
"""


def boot(page_title: str, heading_text: str, blurb: str) -> None:
    st.set_page_config(page_title=page_title, layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-banner"><div class="hero">{heading_text}</div>'
        f'<p class="blurb">{blurb}</p></div>',
        unsafe_allow_html=True,
    )


def heading(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="chart-head">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="chart-sub">{sub}</p>', unsafe_allow_html=True)


def kpis(items: list[dict]) -> None:
    cards = []
    for item in items:
        tone = item.get("tone", "lime")
        hint = f'<div class="kpi-hint">{item["hint"]}</div>' if item.get("hint") else ""
        cards.append(
            f'<div class="kpi {tone}"><div class="kpi-label">{item["label"]}</div>'
            f'<div class="kpi-value">{item["value"]}</div>{hint}</div>'
        )
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def chips(items: list[dict]) -> None:
    bits = []
    for item in items:
        bits.append(
            f'<div class="chip"><div class="t">{item["label"]}</div>'
            f'<div class="v">{item["value"]}</div></div>'
        )
    st.markdown(f'<div class="chip-row">{"".join(bits)}</div>', unsafe_allow_html=True)


def pretty_when(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%b %d, %H:%M UTC")


def _layout(fig: go.Figure, x_title: str, y_title: str, height: int = 430, time_axis: bool = False) -> go.Figure:
    xaxis = dict(
        title=dict(text=f"<b>{x_title}</b>", font=dict(size=17, color=INK, family=FONT)),
        tickfont=dict(size=14, color=MUTED, family=FONT),
        gridcolor=GRID,
        zeroline=False,
        showline=False,
    )
    if time_axis:
        xaxis["tickformat"] = "%b %d\n%H:%M"
    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, color=INK, size=15),
        margin=dict(l=80, r=28, t=18, b=80),
        height=height,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=14, family=FONT)),
        xaxis=xaxis,
        yaxis=dict(
            title=dict(text=f"<b>{y_title}</b>", font=dict(size=17, color=INK, family=FONT)),
            tickfont=dict(size=14, color=MUTED, family=FONT),
            gridcolor=GRID,
            zeroline=False,
            showline=False,
        ),
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


def area_line(
    frame: pd.DataFrame,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    color: str = LIME,
    y_format: str = ".0f",
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="lines",
            line=dict(color=color, width=3, shape="hv" if y == "rate_usd_per_kwh" else "spline"),
            fill="tozeroy",
            fillcolor={
                LIME: "rgba(200,255,61,0.16)",
                CYAN: "rgba(92,225,255,0.16)",
                VIOLET: "rgba(196,181,253,0.16)",
            }.get(color, "rgba(200,255,61,0.16)"),
            hovertemplate="Hour: %{x|%b %d, %H:%M UTC}<br>"
            + f"{y_title}: %{{y:{y_format}}}<extra></extra>",
            showlegend=False,
        )
    )
    show(_layout(fig, x_title, y_title, time_axis=True))


def band_bars(frame: pd.DataFrame, x: str, y: str, band: str, x_title: str, y_title: str) -> None:
    colors = [BAND_COLOR.get(str(b), CYAN) for b in frame[band]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame[x],
            y=frame[y],
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="Hour: %{x|%b %d, %H:%M UTC}<br>Demand: %{y:.0f} MW<extra></extra>",
            showlegend=False,
        )
    )
    show(_layout(fig, x_title, y_title, time_axis=True))


def scatter(frame: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, color_col: str | None = None) -> None:
    fig = go.Figure()
    marker = dict(size=14, color=CYAN, line=dict(width=0), opacity=0.9)
    if color_col and color_col in frame:
        marker = dict(
            size=16,
            color=frame[color_col],
            colorscale=[[0, CYAN], [0.5, LIME], [1, PINK]],
            showscale=False,
            opacity=0.92,
        )
    custom = None
    hover = f"{x_title}: %{{x:.1f}}<br>{y_title}: %{{y:.0f}}<extra></extra>"
    if "ts_utc" in frame.columns:
        custom = [[pretty_when(ts)] for ts in frame["ts_utc"]]
        hover = (
            "Hour: %{customdata[0]}<br>"
            f"{x_title}: %{{x:.1f}}<br>"
            f"{y_title}: %{{y:.0f}}<extra></extra>"
        )
    fig.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="markers",
            marker=marker,
            customdata=custom,
            hovertemplate=hover,
            showlegend=False,
        )
    )
    show(_layout(fig, x_title, y_title))


def hbar(labels: list[str], values: list[float], x_title: str, y_title: str, signed: bool = False) -> None:
    colors = [LIME if val >= 0 else PINK for val in values] if signed else [VIOLET] * len(values)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>Effect: %{x:.1f} MW<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    show(_layout(fig, x_title, y_title, height=max(420, 32 * len(labels) + 90)))


def friendly_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))
