"""
visualization.py
----------------
Driver-side charts (matplotlib / seaborn / plotly).

Visual analytics are intentionally single-node: plot libraries consume the
already-aggregated CSVs / pandas frames produced by Spark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend so main.py works on servers / CI without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False


def _output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_daily_trend(daily_pdf: pd.DataFrame, save: bool = True) -> Path | None:
    """Line chart of system-wide daily consumption."""
    _style()
    fig, ax = plt.subplots()
    d = daily_pdf.copy()
    d["date"] = pd.to_datetime(d["date"])
    ax.plot(d["date"], d["total_kwh"], color="#1f77b4", linewidth=2, label="Total kWh")
    ax.fill_between(d["date"], d["total_kwh"], alpha=0.15, color="#1f77b4")
    ax.set_title("Daily Electricity Consumption Trend (All Areas)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total consumption (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path = _output_dir() / "chart_daily_trend.png"
    if save:
        fig.savefig(path, dpi=140)
        print(f"[Viz] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_area_bar(area_rank_pdf: pd.DataFrame, save: bool = True) -> Path | None:
    """Bar chart of area-wise total consumption."""
    _style()
    fig, ax = plt.subplots()
    data = area_rank_pdf.sort_values("total_kwh", ascending=False)
    palette = sns.color_palette("viridis", n_colors=len(data))
    sns.barplot(data=data, x="area", y="total_kwh", hue="area",
                palette=palette, dodge=False, ax=ax, legend=False)
    ax.set_title("Area-wise Total Electricity Consumption")
    ax.set_xlabel("Area")
    ax.set_ylabel("Total consumption (kWh)")
    for i, row in enumerate(data.itertuples()):
        ax.text(i, row.total_kwh, f"{row.total_kwh:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = _output_dir() / "chart_area_bar.png"
    if save:
        fig.savefig(path, dpi=140)
        print(f"[Viz] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_forecast(
    history_pdf: pd.DataFrame,
    forecast_pdf: pd.DataFrame,
    title: str = "7-Day Consumption Forecast",
    filename: str = "chart_forecast.png",
    save: bool = True,
) -> Path | None:
    """History + forecast overlay."""
    _style()
    fig, ax = plt.subplots()
    h = history_pdf.copy()
    h["date"] = pd.to_datetime(h["date"])
    f = forecast_pdf.copy()
    f["date"] = pd.to_datetime(f["date"])

    ax.plot(h["date"], h["total_kwh"], color="#1f77b4", label="Historical", linewidth=2)
    ycol = "forecast_kwh" if "forecast_kwh" in f.columns else "total_kwh"
    ax.plot(f["date"], f[ycol], color="#d62728", linestyle="--", marker="o",
            label="Forecast", linewidth=2)
    # Bridge line from last history point to first forecast
    if len(h) and len(f):
        ax.plot(
            [h["date"].iloc[-1], f["date"].iloc[0]],
            [h["total_kwh"].iloc[-1], f[ycol].iloc[0]],
            color="#d62728", linestyle=":", linewidth=1.5,
        )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path = _output_dir() / filename
    if save:
        fig.savefig(path, dpi=140)
        print(f"[Viz] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_distribution(raw_or_clean_pdf: pd.DataFrame, save: bool = True) -> Path | None:
    """Histogram + KDE of consumption values (optionally by area)."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    col = "consumption_kwh" if "consumption_kwh" in raw_or_clean_pdf.columns else "total_kwh"
    sns.histplot(raw_or_clean_pdf[col].dropna(), kde=True, ax=axes[0], color="#2ca02c")
    axes[0].set_title("Consumption Distribution")
    axes[0].set_xlabel(col)

    if "area" in raw_or_clean_pdf.columns:
        sns.boxplot(data=raw_or_clean_pdf, x="area", y=col, ax=axes[1],
                    hue="area", palette="Set2", legend=False)
        axes[1].set_title("Consumption by Area")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        sns.violinplot(y=raw_or_clean_pdf[col].dropna(), ax=axes[1], color="#ff7f0e")
        axes[1].set_title("Consumption Violin")

    fig.tight_layout()
    path = _output_dir() / "chart_distribution.png"
    if save:
        fig.savefig(path, dpi=140)
        print(f"[Viz] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_area_forecast_facet(area_fc: pd.DataFrame, save: bool = True) -> Path | None:
    """Small-multiples style per-area forecast lines."""
    if area_fc.empty:
        return None
    _style()
    areas = sorted(area_fc["area"].unique())
    n = len(areas)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows), sharex=False)
    axes = np.array(axes).reshape(-1)
    for i, area in enumerate(areas):
        sub = area_fc[area_fc["area"] == area].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        axes[i].plot(sub["date"], sub["forecast_kwh"], marker="o", color="#9467bd")
        axes[i].set_title(area)
        axes[i].set_ylabel("kWh")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Per-Area 7-Day Forecasts", y=1.02, fontsize=16)
    fig.tight_layout()
    path = _output_dir() / "chart_area_forecasts.png"
    if save:
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"[Viz] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_plotly_interactive(
    daily_pdf: pd.DataFrame,
    forecast_pdf: pd.DataFrame,
    save: bool = True,
) -> Path | None:
    """Optional interactive HTML chart (Plotly)."""
    if not HAS_PLOTLY:
        print("[Viz] plotly not installed - skipping interactive chart")
        return None

    h = daily_pdf.copy()
    h["date"] = pd.to_datetime(h["date"])
    f = forecast_pdf.copy()
    f["date"] = pd.to_datetime(f["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h["date"], y=h["total_kwh"], mode="lines", name="Historical",
        line=dict(color="#1f77b4", width=2),
    ))
    ycol = "forecast_kwh" if "forecast_kwh" in f.columns else "total_kwh"
    fig.add_trace(go.Scatter(
        x=f["date"], y=f[ycol], mode="lines+markers", name="Forecast",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Interactive Daily Consumption + 7-Day Forecast",
        xaxis_title="Date",
        yaxis_title="kWh",
        template="plotly_white",
        hovermode="x unified",
    )
    path = _output_dir() / "chart_forecast_interactive.html"
    if save:
        fig.write_html(str(path))
        print(f"[Viz] Saved {path}")
    return path if save else None


def generate_all_visualizations(
    daily_pdf: pd.DataFrame,
    area_rank_pdf: pd.DataFrame,
    forecast_better: pd.DataFrame,
    area_forecast: pd.DataFrame,
    clean_sample_pdf: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Create the full chart suite used by main.py and the dashboard."""
    print("\n" + "=" * 70)
    print(" VISUALIZATION")
    print("=" * 70)

    paths = {
        "daily_trend": plot_daily_trend(daily_pdf),
        "area_bar": plot_area_bar(area_rank_pdf),
        "forecast": plot_forecast(
            daily_pdf, forecast_better,
            title="7-Day Forecast (ARIMA / Linear)",
            filename="chart_forecast.png",
        ),
        "area_forecasts": plot_area_forecast_facet(area_forecast),
        "plotly": plot_plotly_interactive(daily_pdf, forecast_better),
    }
    if clean_sample_pdf is not None and not clean_sample_pdf.empty:
        paths["distribution"] = plot_distribution(clean_sample_pdf)
    else:
        # Fall back to daily totals distribution
        paths["distribution"] = plot_distribution(
            daily_pdf.rename(columns={"total_kwh": "consumption_kwh"})
        )

    return paths
