"""
Load Forecasting Dashboard: Power Consumption of Tetouan City (UCI dataset 849)
Zone 1 / Zone 2 / Zone 3 distribution networks, 168-hour (7-day) hourly forecast.

Run:  streamlit run app/streamlit_app.py   (from the electricity-prediction/ folder)

The app reads the precomputed 7-day forecast (data/processed/future_7day_forecast.csv,
produced by src/make_deployment_forecast.py) plus the cleaned history, rather than
rebuilding the feature pipeline at runtime.

It shows the seasonal-naive forecast alongside the model, because on this dataset the
seasonal naive is the more accurate of the two. Presenting the model alone would imply
a level of skill the evaluation does not support. See reports/05_modeling_report.md.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FORECAST_CSV = ROOT / "data" / "processed" / "future_7day_forecast.csv"
HISTORY_CSV = ROOT / "data" / "processed" / "cleaned_energy_data.csv"

# Per-zone metrics from reports/results_corrected.json. Test window is untouched
# until final scoring; walk-forward covers three non-overlapping 42-day windows.
ZONES = {
    "Zone 1": {
        "col": "PowerConsumption_Zone1",
        "color": "#4477AA",
        "model": "Random Forest",
        "test_mape": 4.20,
        "snaive_mape": 2.86,
        "persistence_mape": 24.72,
        "windows": [4.52, 6.19, 6.32],
        "windows_snaive": [2.86, 4.39, 5.64],
        "desc": "Highest and most stable load, with an essentially flat annual trend. "
                "The most repeatable of the three networks week to week.",
    },
    "Zone 2": {
        "col": "PowerConsumption_Zone2",
        "color": "#228833",
        "model": "Random Forest",
        "test_mape": 4.81,
        "snaive_mape": 4.26,
        "persistence_mape": 28.60,
        "windows": [4.79, 5.53, 9.36],
        "windows_snaive": [4.26, 5.06, 9.11],
        "desc": "Strongest weekday effect and a genuine rising trend across the year "
                "of about 28 percent.",
    },
    "Zone 3": {
        "col": "PowerConsumption_Zone3",
        "color": "#EE6677",
        "model": "XGBoost",
        "test_mape": 20.64,
        "snaive_mape": 8.73,
        "persistence_mape": 28.35,
        "windows": [22.10, 9.09, 18.11],
        "windows_snaive": [8.73, 10.08, 10.92],
        "desc": "Non-stationary and most volatile, with a steep annual decline and a "
                "sharp July to August surge. The model performs worst here by a wide margin.",
    },
}

C_SNAIVE = "#EE6677"
C_HISTORY = "#666666"
HISTORY_DAYS_DEFAULT = 7

st.set_page_config(page_title="Tetouan Load Forecast", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def load_forecast():
    return pd.read_csv(FORECAST_CSV, parse_dates=["Datetime"]).set_index("Datetime")


@st.cache_data(show_spinner=False)
def load_history():
    cols = ["Datetime"] + [z["col"] for z in ZONES.values()]
    df = pd.read_csv(HISTORY_CSV, usecols=cols, parse_dates=["Datetime"])
    df = df.set_index("Datetime").resample("h").mean()
    return df.rename(columns={z["col"]: name for name, z in ZONES.items()})


try:
    forecast = load_forecast()
    history = load_history()
except FileNotFoundError as exc:
    st.error(
        f"Could not find a required data file:\n\n`{exc.filename}`\n\n"
        "Run `python src/data_cleaning.py`, `python src/run_pipeline.py 300` and "
        "`python src/make_deployment_forecast.py` first, and launch Streamlit from "
        "the `electricity-prediction/` folder."
    )
    st.stop()

with st.sidebar:
    st.title("⚡ Tetouan Load Forecast")
    st.caption("7-day forecast, Zone 1 / Zone 2 / Zone 3")
    zone_name = st.radio("Select zone", list(ZONES.keys()), index=0)
    meta = ZONES[zone_name]

    st.divider()
    hist_days = st.slider("History shown before forecast (days)", 0, 14, HISTORY_DAYS_DEFAULT)
    st.divider()
    st.caption(
        f"**Model:** {meta['model']}  \n"
        f"**Test MAPE:** {meta['test_mape']:.2f}%  \n"
        f"**Seasonal naive:** {meta['snaive_mape']:.2f}%  \n"
        "UCI Tetouan City dataset, 2017"
    )

fc = forecast[zone_name]
fc_snaive = forecast.get(f"{zone_name} seasonal naive")
hist = history[zone_name]
hist_window = hist.iloc[-hist_days * 24:] if hist_days > 0 else hist.iloc[0:0]
daily_totals = fc.resample("D").sum() / 1000.0
fc_start, fc_end = fc.index[0], fc.index[-1]

st.markdown(f"## {zone_name}, 7-day energy forecast")
st.caption(
    f"Horizon: **{fc_start:%d %b %Y}** to **{fc_end:%d %b %Y}** (168 hours ahead) "
    f"· Model: **{meta['model']}**"
)

st.warning(
    f"On the held-out test window the **seasonal naive baseline is more accurate than "
    f"the model** for this zone ({meta['snaive_mape']:.2f}% vs {meta['test_mape']:.2f}% MAPE). "
    "Both forecasts are shown below. See the Accuracy tab.",
    icon="⚠️",
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Peak load (model)", f"{fc.max():,.0f} kW", help="Highest single hour forecast")
k2.metric("Average load (model)", f"{fc.mean():,.0f} kW")
k3.metric("Total energy, 7 days", f"{fc.sum()/1000:,.1f} MWh")
k4.metric("Model vs seasonal naive", f"{meta['test_mape'] - meta['snaive_mape']:+.2f} pp",
          help="Difference in test MAPE. Positive means the model is worse than the baseline.",
          delta=None)

tab_fc, tab_daily, tab_acc, tab_about = st.tabs(
    ["📈 Hourly forecast", "📊 Daily totals", "🎯 Accuracy", "ℹ️ About"])

with tab_fc:
    fig = go.Figure()
    if hist_days > 0:
        fig.add_trace(go.Scatter(
            x=hist_window.index, y=hist_window.values, mode="lines", name="Observed history",
            line=dict(color=C_HISTORY, width=1.4),
            hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:,.0f} kW<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc.values, mode="lines", name=f"Model ({meta['model']})",
        line=dict(color=meta["color"], width=2.2),
        hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:,.0f} kW<extra></extra>"))
    if fc_snaive is not None:
        fig.add_trace(go.Scatter(
            x=fc_snaive.index, y=fc_snaive.values, mode="lines",
            name="Seasonal naive (more accurate)",
            line=dict(color=C_SNAIVE, width=1.8, dash="dot"),
            hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:,.0f} kW<extra></extra>"))
    if hist_days > 0:
        fig.add_vline(x=fc_start, line_dash="dot", line_color="#999999")
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Power (kW)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Weather inputs use values observed exactly one week before each target hour, "
        "so no external weather forecast is required. No actuals exist for this period; "
        "it demonstrates the deployment path rather than measuring accuracy."
    )

with tab_daily:
    fig2 = go.Figure(go.Bar(
        x=[d.strftime("%a %d %b") for d in daily_totals.index], y=daily_totals.values,
        marker_color=meta["color"], hovertemplate="%{x}<br>%{y:,.1f} MWh<extra></extra>"))
    fig2.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10),
                       yaxis_title="Energy (MWh/day)")
    st.plotly_chart(fig2, width="stretch")
    st.dataframe(pd.DataFrame({
        "Day": [d.strftime("%A, %d %b %Y") for d in daily_totals.index],
        "Total energy (MWh)": daily_totals.round(2).values,
        "Peak hour (kW)": fc.resample("D").max().round(0).values,
        "Avg load (kW)": fc.resample("D").mean().round(0).values,
    }), width="stretch", hide_index=True)

with tab_acc:
    st.markdown("#### How accurate is this model?")
    st.write(
        f"For **{zone_name}**, the {meta['model']} model was scored once on six weeks of "
        "held-out data, and stress-tested on three separate 42-day windows. Lower MAPE is better."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", f"{meta['test_mape']:.2f}%")
    c2.metric("Seasonal naive", f"{meta['snaive_mape']:.2f}%",
              help="Predict the value at the same hour one week earlier.")
    c3.metric("Persistence", f"{meta['persistence_mape']:.2f}%",
              help="Predict the value observed at forecast time.")

    st.markdown("**Walk-forward validation**, three non-overlapping 42-day windows:")
    st.dataframe(pd.DataFrame({
        "Window": ["W1 (Nov to Dec)", "W2 (Oct to Nov)", "W3 (Aug to Oct)"],
        "Model MAPE (%)": meta["windows"],
        "Seasonal naive MAPE (%)": meta["windows_snaive"],
        "Model wins": [m < s for m, s in zip(meta["windows"], meta["windows_snaive"])],
    }), width="stretch", hide_index=True)

    st.info(
        "Across all three zones and all three windows, the model beats the seasonal naive in "
        "**1 of 9** combinations, and that one is the window its hyperparameters were tuned on. "
        "The honest conclusion is that this model does not outperform a seasonal naive at a "
        "168-hour horizon on this dataset. An earlier version of this project reported the "
        "opposite because of a baseline indexing defect, now fixed and documented.",
        icon="ℹ️",
    )

    st.divider()
    st.markdown("**All zones, headline test MAPE:**")
    st.dataframe(pd.DataFrame({
        "Zone": list(ZONES.keys()),
        "Model": [z["model"] for z in ZONES.values()],
        "Model MAPE (%)": [z["test_mape"] for z in ZONES.values()],
        "Seasonal naive (%)": [z["snaive_mape"] for z in ZONES.values()],
        "Persistence (%)": [z["persistence_mape"] for z in ZONES.values()],
    }), width="stretch", hide_index=True)

with tab_about:
    st.markdown("#### About this dashboard")
    st.write(meta["desc"])
    st.markdown(
        """
Forecasts hourly electricity consumption for three urban distribution networks in
**Tetouan, Morocco** over a **7-day (168-hour) horizon**.

**Method**
- One model per zone, family and hyperparameters chosen by FLAML AutoML on a
  validation split only, then refit on train plus validation and scored once on test.
- Inputs: the zone's own past consumption (lags at 24, 168 and 336 hours, plus rolling
  statistics), calendar features, and weather from one week before each target hour.
- Chronological split with a 168-hour embargo so no split's targets reach into the next.
  No random splitting, which would leak the future.

**Data**
- Power Consumption of Tetouan City, UCI Machine Learning Repository dataset 849.
  Salam and El Hibaoui, IEEE IRSEC 2018. 10-minute readings for 2017, aggregated hourly.
- Forecast shown: 31 Dec 2017 to 6 Jan 2018, from models retrained on the full year.

**Correctness**
- `python src/validate_project.py` runs 40 assertions covering feature observability,
  embargo integrity, baseline definitions, effective sample size and metric reproducibility.

Full methodology and limitations are in `reports/05_modeling_report.md`.
        """
    )
    st.caption("MS application portfolio project. Methodology in reports/05_modeling_report.md")
