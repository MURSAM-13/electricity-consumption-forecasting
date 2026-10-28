"""
dashboard.py
------------
Interactive Streamlit dashboard for electricity consumption insights.

Run standalone:
  streamlit run src/dashboard.py

Or from main.py:
  python main.py --dashboard

Reads pre-computed CSVs from output/ so the dashboard does not need a live
SparkSession (fast reload while demoing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path when launched via `streamlit run src/dashboard.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUT = _PROJECT_ROOT / "output"


def _load_csv(name: str) -> pd.DataFrame:
    path = OUTPUT / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "daily": _load_csv("daily_totals.csv"),
        "area_rank": _load_csv("area_ranking.csv"),
        "peak": _load_csv("peak_day.csv"),
        "stats": _load_csv("summary_stats.csv"),
        "area_daily": _load_csv("area_daily.csv"),
        "fc_better": _load_csv("forecast_overall_better.csv"),
        "fc_trend": _load_csv("forecast_overall_trend.csv"),
        "fc_area": _load_csv("forecast_per_area.csv"),
    }


def build_dashboard() -> None:
    """Streamlit app entry (called when __name__ == '__main__' or from main)."""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(
        page_title="Electricity Consumption Analysis & Forecasting",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(" Electricity Consumption Analysis & Forecasting")
    st.caption(
        "B.Tech Big Data Computing Technologies · PySpark analytics + Python forecasting"
    )

    data = load_all()
    missing = [k for k, v in data.items() if v.empty and k in ("daily", "area_rank")]
    if missing:
        st.error(
            "Required analysis outputs not found in `output/`. "
            "Run the pipeline first:\n\n```\npython main.py\n```"
        )
        st.stop()

    daily = data["daily"]
    area_rank = data["area_rank"]
    area_daily = data["area_daily"]
    fc_better = data["fc_better"]
    fc_trend = data["fc_trend"]
    fc_area = data["fc_area"]
    peak = data["peak"]
    stats = data["stats"]

    # ---- Sidebar filters -------------------------------------------------------
    st.sidebar.header("Filters")
    areas = sorted(area_rank["area"].dropna().unique().tolist()) if not area_rank.empty else []
    selected_areas = st.sidebar.multiselect(
        "Areas",
        options=areas,
        default=areas,
        help="Filter charts and tables by service area",
    )
    if not selected_areas:
        selected_areas = areas

    show_trend_method = st.sidebar.radio(
        "Overall forecast method",
        options=["ARIMA / better model", "Simple trend"],
        index=0,
    )
    fc_overall = fc_better if show_trend_method.startswith("ARIMA") else fc_trend

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Big Data stack demo**  \n"
        "- Ingestion: Spark CSV (local / HDFS)  \n"
        "- ETL: distributed DataFrame ops  \n"
        "- Analytics: Spark SQL ~ HiveQL  \n"
        "- Forecast: pandas + sklearn / ARIMA  \n"
        "- UI: Streamlit"
    )

    # ---- Summary cards ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    total_kwh = float(daily["total_kwh"].sum()) if not daily.empty else 0.0
    avg_daily = float(daily["total_kwh"].mean()) if not daily.empty else 0.0
    n_days = int(daily["date"].nunique()) if not daily.empty else 0
    top_area = (
        area_rank.sort_values("total_kwh", ascending=False).iloc[0]["area"]
        if not area_rank.empty else "-"
    )
    peak_day_str = "-"
    peak_val = 0.0
    if not peak.empty:
        peak_day_str = str(pd.to_datetime(peak.iloc[0]["date"]).date())
        peak_val = float(peak.iloc[0]["total_kwh"])

    c1.metric("Total consumption", f"{total_kwh:,.0f} kWh")
    c2.metric("Avg daily total", f"{avg_daily:,.0f} kWh")
    c3.metric("Days covered", f"{n_days}")
    c4.metric("Top area", top_area)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Peak day", peak_day_str)
    c6.metric("Peak load", f"{peak_val:,.0f} kWh")
    if not stats.empty:
        c7.metric("Record mean", f"{float(stats.iloc[0].get('mean', 0)):,.0f} kWh")
        c8.metric("Record stddev", f"{float(stats.iloc[0].get('stddev', 0)):,.0f} kWh")
    else:
        c7.metric("Record mean", "-")
        c8.metric("Record stddev", "-")

    st.markdown("---")

    # ---- Area comparison -------------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("Area comparison")
        ar = area_rank[area_rank["area"].isin(selected_areas)].copy()
        fig_bar = px.bar(
            ar.sort_values("total_kwh", ascending=False),
            x="area",
            y="total_kwh",
            color="area",
            text_auto=".2s",
            title="Total consumption by area",
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(
            ar.sort_values("rank") if "rank" in ar.columns else ar,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Daily trend")
        if not area_daily.empty:
            ad = area_daily[area_daily["area"].isin(selected_areas)].copy()
            # Also show system total
            fig_line = px.line(
                ad,
                x="date",
                y="total_kwh",
                color="area",
                title="Daily consumption by area",
            )
            fig_line.update_layout(height=400, legend_title_text="Area")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            fig_line = px.line(daily, x="date", y="total_kwh", title="Daily total")
            st.plotly_chart(fig_line, use_container_width=True)

    # ---- System daily + forecast ----------------------------------------------
    st.subheader("System-wide history & 7-day forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["total_kwh"],
        mode="lines", name="Historical",
        line=dict(color="#1f77b4", width=2),
    ))
    if not fc_overall.empty:
        ycol = "forecast_kwh" if "forecast_kwh" in fc_overall.columns else "total_kwh"
        fig.add_trace(go.Scatter(
            x=fc_overall["date"], y=fc_overall[ycol],
            mode="lines+markers", name="Forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
        ))
    fig.update_layout(
        height=420,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="kWh",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    fc_col1, fc_col2 = st.columns(2)
    with fc_col1:
        st.markdown("**Overall forecast table**")
        st.dataframe(fc_overall, use_container_width=True, hide_index=True)
    with fc_col2:
        st.markdown("**Per-area forecast (filtered)**")
        if not fc_area.empty:
            fa = fc_area[fc_area["area"].isin(selected_areas)]
            st.dataframe(fa, use_container_width=True, hide_index=True)
            fig_a = px.line(
                fa, x="date", y="forecast_kwh", color="area",
                markers=True, title="Per-area 7-day forecast",
            )
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("Per-area forecast CSV not found.")

    # ---- Distribution ----------------------------------------------------------
    st.subheader("Distribution")
    if not area_daily.empty:
        ad2 = area_daily[area_daily["area"].isin(selected_areas)]
        fig_h = px.histogram(
            ad2, x="total_kwh", color="area", nbins=30, marginal="box",
            title="Daily area totals distribution",
            barmode="overlay", opacity=0.65,
        )
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Outputs generated by the PySpark pipeline · "
        f"data folder: `{OUTPUT}`"
    )


def launch_dashboard() -> None:
    """
    Helper used by main.py. Streamlit apps should be started via the
    `streamlit run` CLI; we spawn that process for convenience.
    """
    import subprocess

    dash_path = Path(__file__).resolve()
    print(f"[Dashboard] Launching Streamlit -> {dash_path}")
    print("            Press Ctrl+C to stop the dashboard server.")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dash_path), "--server.headless", "true"],
        check=False,
    )


if __name__ == "__main__":
    build_dashboard()
