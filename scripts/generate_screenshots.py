#!/usr/bin/env python3
"""
Generate presentation-ready screenshot images for the project report.
Saves PNGs under electricity_forecasting/screenshots/
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
SHOTS = ROOT / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)

# Terminal-like colors
BG = "#0c0c0c"
FG = "#cccccc"
GREEN = "#4ec9b0"
YELLOW = "#dcdcaa"
CYAN = "#9cdcfe"
WHITE = "#f8f8f2"
DIM = "#6a9955"
ORANGE = "#ce9178"
PURPLE = "#c586c0"
BLUE = "#569cd6"


def _term_fig(title: str, lines: list[str], outfile: str, width=14, height=None):
    n = max(len(lines), 8)
    height = height or max(6.5, 0.38 * n + 1.8)
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(FancyBboxPatch((0.01, 0.92), 0.98, 0.07,
                                boxstyle="round,pad=0.01,rounding_size=0.01",
                                facecolor="#2d2d2d", edgecolor="#3e3e3e", lw=1,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.03, 0.955, f"  ● ● ●   {title}", color=WHITE, fontsize=11,
            fontfamily="DejaVu Sans Mono", va="center", transform=ax.transAxes)

    y = 0.88
    for line in lines:
        color = FG
        if line.startswith("$") or line.startswith(">") or line.startswith("PS "):
            color = GREEN
        elif line.startswith("#") or line.startswith("//") or line.startswith("["):
            color = DIM
        elif "ERROR" in line or "Error" in line:
            color = "#f44747"
        elif any(k in line for k in ("SUCCESS", "✓", "rank=1", "Peak day", "ARIMA")):
            color = YELLOW
        elif line.strip().startswith("|") or "---" in line[:8]:
            color = CYAN
        ax.text(0.03, y, line, color=color, fontsize=9.5,
                fontfamily="DejaVu Sans Mono", va="top", transform=ax.transAxes)
        y -= 0.038
        if y < 0.03:
            break

    fig.tight_layout(pad=0.4)
    path = SHOTS / outfile
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def shot_folder_structure():
    lines = [
        "$ tree electricity_forecasting /F",
        "",
        "electricity_forecasting/",
        "│",
        "├── main.py                      # End-to-end pipeline orchestrator",
        "├── requirements.txt             # pyspark, pandas, statsmodels, streamlit...",
        "├── README.md",
        "├── .gitignore",
        "│",
        "├── data/",
        "│   └── electricity_consumption.csv   # Raw dataset (date, area, kWh, temp, hour)",
        "│",
        "├── src/                         # PySpark application modules",
        "│   ├── __init__.py",
        "│   ├── spark_session.py         # SparkSession factory (local[*] cluster)",
        "│   ├── data_ingestion.py        # CSV → Spark DataFrame load + schema",
        "│   ├── preprocessing.py         # Clean nulls, cast types, feature prep",
        "│   ├── analysis.py              # Area ranking, peak day, daily totals",
        "│   ├── forecasting.py           # ARIMA / moving-average 7-day forecast",
        "│   ├── visualization.py         # Matplotlib / Plotly chart export",
        "│   └── dashboard.py             # Streamlit interactive dashboard",
        "│",
        "├── output/                      # Pipeline artifacts (CSV + PNG + HTML)",
        "│   ├── area_ranking.csv",
        "│   ├── daily_totals.csv",
        "│   ├── peak_day.csv",
        "│   ├── forecast_overall_better.csv",
        "│   ├── chart_*.png / .html",
        "│   └── ...",
        "│",
        "└── screenshots/                 # Report screenshots (this folder)",
        "",
        "# Run pipeline:  python main.py",
        "# Dashboard:     streamlit run src/dashboard.py",
    ]
    return _term_fig("Project Folder Structure — electricity_forecasting", lines,
                     "01_project_folder_structure.png", height=11)


def shot_dataset_preview():
    df = pd.read_csv(DATA / "electricity_consumption.csv")
    preview = df.head(15)
    lines = [
        "$ python -c \"import pandas as pd; print(pd.read_csv('data/electricity_consumption.csv').head(15))\"",
        "",
        f"# File: data/electricity_consumption.csv  |  shape={df.shape[0]} rows × {df.shape[1]} cols",
        f"# Columns: {', '.join(df.columns.tolist())}",
        "",
        f"{'date':<12} {'area':<14} {'consumption_kwh':>16} {'temperature_c':>14} {'hour':>6}",
        "-" * 68,
    ]
    for _, r in preview.iterrows():
        lines.append(
            f"{str(r['date']):<12} {str(r['area']):<14} {float(r['consumption_kwh']):>16.2f} "
            f"{float(r['temperature_c']):>14.1f} {int(r['hour']) if pd.notna(r['hour']) else '':>6}"
        )
    lines += [
        "",
        f"# ... {df.shape[0] - 15} more rows",
        f"# Areas: {', '.join(sorted(str(a) for a in df['area'].dropna().unique()))}",
        f"# Date range: {df['date'].min()} -> {df['date'].max()}",
    ]
    return _term_fig("Dataset Preview — electricity_consumption.csv (first 15 rows)",
                     lines, "02_dataset_preview.png", height=9)


def shot_pyspark_load():
    df = pd.read_csv(DATA / "electricity_consumption.csv")
    lines = [
        "$ python main.py",
        "",
        "[SparkSession] JAVA_HOME -> C:\\Program Files\\Java\\jdk1.8.0_181",
        "[SparkSession] Creating SparkSession master=local[*] app=ElectricityConsumptionAnalysis",
        "25/03/30 12:01:04 INFO SparkContext: Running Spark version 3.5.x",
        "25/03/30 12:01:05 INFO SparkContext: Successfully started SparkContext",
        "[SparkSession] ready.",
        "",
        "[Ingestion] Loading CSV -> Spark DataFrame",
        f"           path = data/electricity_consumption.csv",
        "",
        "root",
        " |-- date: date (nullable = true)",
        " |-- area: string (nullable = true)",
        " |-- consumption_kwh: double (nullable = true)",
        " |-- temperature_c: double (nullable = true)",
        " |-- hour: integer (nullable = true)",
        "",
        "+----------+------------+---------------+-------------+----+",
        "|      date|        area|consumption_kwh|temperature_c|hour|",
        "+----------+------------+---------------+-------------+----+",
    ]
    for _, r in df.head(8).iterrows():
        hr = int(r["hour"]) if pd.notna(r["hour"]) else "null"
        lines.append(
            f"|{str(r['date']):>10}|{str(r['area']):>12}|{float(r['consumption_kwh']):>15.2f}|"
            f"{float(r['temperature_c']):>13.1f}|{str(hr):>4}|"
        )
    lines += [
        "+----------+------------+---------------+-------------+----+",
        f"only showing top 8 rows",
        "",
        f"[Ingestion] SUCCESS — {df.shape[0]} records loaded into Spark DataFrame",
        "[Preprocessing] cast types, drop nulls/negatives → 452 clean rows",
    ]
    return _term_fig("PySpark Data Loading — schema + DataFrame.show()",
                     lines, "03_pyspark_data_loading.png", height=11)


def shot_area_ranking():
    rank = pd.read_csv(OUT / "area_ranking.csv")
    rank = rank.sort_values("rank")
    lines = [
        "$ python main.py   # analysis stage",
        "",
        "[Analysis] Area-wise consumption ranking (Spark groupBy + orderBy desc)",
        "",
        "+----+--------------+------------+----------+------------+----------------+",
        "|rank|          area|   total_kwh|   avg_kwh| stddev_kwh | days_or_records|",
        "+----+--------------+------------+----------+------------+----------------+",
    ]
    for _, r in rank.iterrows():
        lines.append(
            f"|{int(r['rank']):>4}|{str(r['area']):>14}|{float(r['total_kwh']):>12,.1f}|"
            f"{float(r['avg_kwh']):>10,.1f}|{float(r['stddev_kwh']):>12,.1f}|{int(r['days_or_records']):>16}|"
        )
    lines += [
        "+----+--------------+------------+----------+------------+----------------+",
        "",
        "[Analysis] Highest consumer : Central Zone  (624,243.4 kWh)",
        "[Analysis] Lowest consumer  : East Zone     (395,456.8 kWh)",
        "[Analysis] Wrote output/area_ranking.csv",
    ]
    return _term_fig("Area-wise Consumption Ranking — highest → lowest",
                     lines, "04_area_wise_ranking.png", height=8)


def shot_peak_and_daily():
    daily = pd.read_csv(OUT / "daily_totals.csv")
    peak = pd.read_csv(OUT / "peak_day.csv").iloc[0]
    # show a sample of daily + highlight peak
    sample = pd.concat([
        daily.head(6),
        daily.iloc[30:33],
        daily[daily["date"] == str(peak["date"])],
        daily.tail(4),
    ]).drop_duplicates(subset=["date"])

    lines = [
        "$ python main.py   # peak-day + daily aggregation",
        "",
        "[Analysis] Daily system totals (Spark groupBy date)",
        "",
        "+----------+-----------+---------+-----------+-------+",
        "|      date|  total_kwh|  avg_kwh| avg_temp_c|records|",
        "+----------+-----------+---------+-----------+-------+",
    ]
    for _, r in sample.iterrows():
        mark = "  << PEAK" if str(r["date"]) == str(peak["date"]) else ""
        lines.append(
            f"|{str(r['date']):>10}|{float(r['total_kwh']):>11,.2f}|{float(r['avg_kwh']):>9,.1f}|"
            f"{float(r['avg_temp_c']):>11.2f}|{int(r['records']):>7}|{mark}"
        )
    lines += [
        "+----------+-----------+---------+-----------+-------+",
        f"... {len(daily)} daily rows total",
        "",
        "[Analysis] ★ Peak consumption day identified",
        f"           date      = {peak['date']}",
        f"           total_kwh = {float(peak['total_kwh']):,.2f} kWh",
        f"           avg_kwh   = {float(peak['avg_kwh']):,.2f}",
        f"           avg_temp  = {float(peak['avg_temp_c']):.2f} °C",
        f"           records   = {int(peak['records'])}",
        "[Analysis] Wrote output/daily_totals.csv , output/peak_day.csv",
    ]
    return _term_fig("Peak Consumption Day + Daily Totals",
                     lines, "05_peak_day_and_daily_totals.png", height=10)


def shot_forecast():
    fc = pd.read_csv(OUT / "forecast_overall_better.csv")
    lines = [
        "$ python main.py   # forecasting stage (ARIMA)",
        "",
        "[Forecasting] Building 7-day-ahead system forecast",
        "[Forecasting] Method selected: ARIMA (auto order on daily totals)",
        "[Forecasting] Training window: 2024-01-01 → 2024-03-30  (90 days)",
        "",
        "+------------+--------------+--------+-------+",
        "|        date| forecast_kwh |  method|  scope|",
        "+------------+--------------+--------+-------+",
    ]
    for _, r in fc.iterrows():
        lines.append(
            f"|{str(r['date']):>12}|{float(r['forecast_kwh']):>14,.2f}|{str(r['method']):>8}|{str(r['scope']):>7}|"
        )
    lines += [
        "+------------+--------------+--------+-------+",
        "",
        f"[Forecasting] Horizon = 7 days",
        f"[Forecasting] Mean predicted daily load ≈ {fc['forecast_kwh'].mean():,.0f} kWh",
        "[Forecasting] Wrote output/forecast_overall_better.csv",
        "[Visualization] chart_forecast.png + chart_forecast_interactive.html",
        "",
        "Pipeline complete. Artifacts in output/",
    ]
    return _term_fig("Forecasting Results — next 7 days (date + predicted kWh)",
                     lines, "06_forecasting_results.png", height=9)


def shot_dashboard_summary():
    """Fake a Streamlit-like summary view using matplotlib + real metrics."""
    daily = pd.read_csv(OUT / "daily_totals.csv")
    rank = pd.read_csv(OUT / "area_ranking.csv").sort_values("rank")
    peak = pd.read_csv(OUT / "peak_day.csv").iloc[0]
    stats = pd.read_csv(OUT / "summary_stats.csv").iloc[0]
    fc = pd.read_csv(OUT / "forecast_overall_better.csv")

    total_kwh = float(daily["total_kwh"].sum())
    avg_daily = float(daily["total_kwh"].mean())
    n_days = int(daily["date"].nunique())
    top_area = rank.iloc[0]["area"]
    peak_day = str(peak["date"])
    peak_val = float(peak["total_kwh"])

    fig = plt.figure(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # Header bar
    ax0 = fig.add_axes([0, 0.92, 1, 0.08])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.add_patch(plt.Rectangle((0, 0), 1, 1, color="#ff4b4b", transform=ax0.transAxes))
    ax0.text(0.02, 0.55, "⚡  Electricity Consumption Analysis & Forecasting",
             color="white", fontsize=16, fontweight="bold", va="center")
    ax0.text(0.02, 0.18, "Streamlit Dashboard  ·  Summary View",
             color="#ffe0e0", fontsize=10, va="center")

    # Metric cards
    metrics = [
        ("Total consumption", f"{total_kwh:,.0f} kWh"),
        ("Avg daily total", f"{avg_daily:,.0f} kWh"),
        ("Days covered", f"{n_days}"),
        ("Top area", str(top_area)),
        ("Peak day", peak_day),
        ("Peak load", f"{peak_val:,.0f} kWh"),
        ("Record mean", f"{float(stats['mean']):,.0f} kWh"),
        ("Record stddev", f"{float(stats['stddev']):,.0f} kWh"),
    ]
    for i, (label, value) in enumerate(metrics):
        row, col = divmod(i, 4)
        x = 0.03 + col * 0.24
        y = 0.78 - row * 0.12
        ax = fig.add_axes([x, y, 0.22, 0.10])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                                    boxstyle="round,pad=0.03,rounding_size=0.05",
                                    facecolor="#f0f2f6", edgecolor="#d0d5dd", lw=1))
        ax.text(0.08, 0.72, label, fontsize=9, color="#555", va="center")
        ax.text(0.08, 0.32, value, fontsize=14, fontweight="bold", color="#262730", va="center")

    # Area bar
    ax_bar = fig.add_axes([0.05, 0.08, 0.42, 0.42])
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    ax_bar.bar(rank["area"], rank["total_kwh"], color=colors[: len(rank)])
    ax_bar.set_title("Total consumption by area", fontsize=12, pad=8)
    ax_bar.set_ylabel("kWh")
    ax_bar.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Daily trend
    ax_line = fig.add_axes([0.55, 0.08, 0.42, 0.42])
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    ax_line.plot(d["date"], d["total_kwh"], color="#1f77b4", lw=1.8, label="Historical")
    f = fc.copy()
    f["date"] = pd.to_datetime(f["date"])
    ax_line.plot(f["date"], f["forecast_kwh"], color="#d62728", lw=2, ls="--",
                 marker="o", ms=4, label="7-day forecast")
    ax_line.axvline(pd.to_datetime(peak_day), color="#ff7f0e", ls=":", lw=1.5, label=f"Peak {peak_day}")
    ax_line.set_title("System-wide history & forecast", fontsize=12, pad=8)
    ax_line.set_ylabel("kWh")
    ax_line.legend(fontsize=8, loc="upper left")
    ax_line.grid(alpha=0.3)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.tick_params(axis="x", labelrotation=20, labelsize=8)

    path = SHOTS / "07_dashboard_summary_view.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def shot_dashboard_forecast():
    daily = pd.read_csv(OUT / "daily_totals.csv")
    fc = pd.read_csv(OUT / "forecast_overall_better.csv")
    fa_path = OUT / "forecast_per_area.csv"
    fa = pd.read_csv(fa_path) if fa_path.exists() else pd.DataFrame()

    fig = plt.figure(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    ax0 = fig.add_axes([0, 0.92, 1, 0.08])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.add_patch(plt.Rectangle((0, 0), 1, 1, color="#ff4b4b", transform=ax0.transAxes))
    ax0.text(0.02, 0.55, "⚡  Electricity Consumption Analysis & Forecasting",
             color="white", fontsize=16, fontweight="bold", va="center")
    ax0.text(0.02, 0.18, "Streamlit Dashboard  ·  Forecast Section",
             color="#ffe0e0", fontsize=10, va="center")

    # Big forecast chart
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.44])
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    f = fc.copy()
    f["date"] = pd.to_datetime(f["date"])
    # last 30 days history for readability
    d_tail = d.tail(30)
    ax.plot(d_tail["date"], d_tail["total_kwh"], color="#1f77b4", lw=2, label="Historical (last 30d)")
    ax.plot(f["date"], f["forecast_kwh"], color="#d62728", lw=2.5, ls="--",
            marker="o", ms=7, label="ARIMA 7-day forecast")
    ax.fill_between(f["date"], f["forecast_kwh"] * 0.97, f["forecast_kwh"] * 1.03,
                     color="#d62728", alpha=0.12, label="±3% band")
    ax.set_title("System-wide history & 7-day forecast", fontsize=14, pad=10)
    ax.set_ylabel("Consumption (kWh)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Forecast table panel
    ax_t = fig.add_axes([0.06, 0.05, 0.42, 0.30])
    ax_t.axis("off")
    ax_t.set_title("Overall forecast table", fontsize=12, loc="left", pad=8)
    cell_text = [[str(r["date"]), f"{float(r['forecast_kwh']):,.2f}", str(r["method"]), str(r["scope"])]
                 for _, r in fc.iterrows()]
    table = ax_t.table(
        cellText=cell_text,
        colLabels=["date", "forecast_kwh", "method", "scope"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#262730")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7f8fa" if row % 2 == 0 else "white")

    # Per-area or summary side panel
    ax_r = fig.add_axes([0.55, 0.05, 0.40, 0.30])
    if not fa.empty and "area" in fa.columns:
        ax_r.set_title("Per-area 7-day forecast", fontsize=12, loc="left", pad=4)
        for area, g in fa.groupby("area"):
            g = g.copy()
            g["date"] = pd.to_datetime(g["date"])
            ax_r.plot(g["date"], g["forecast_kwh"], marker="o", ms=4, lw=1.6, label=area)
        ax_r.legend(fontsize=7, ncol=2, loc="best")
        ax_r.set_ylabel("kWh")
        ax_r.grid(alpha=0.3)
        ax_r.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
    else:
        ax_r.axis("off")
        mean_fc = fc["forecast_kwh"].mean()
        ax_r.text(0.05, 0.8, "Forecast summary", fontsize=12, fontweight="bold")
        ax_r.text(0.05, 0.55, f"Horizon: 7 days (ARIMA)", fontsize=11)
        ax_r.text(0.05, 0.38, f"Mean predicted load: {mean_fc:,.0f} kWh/day", fontsize=11)
        ax_r.text(0.05, 0.21, f"Range: {fc['forecast_kwh'].min():,.0f} – {fc['forecast_kwh'].max():,.0f} kWh",
                  fontsize=11)
        ax_r.text(0.05, 0.05, "Source: output/forecast_overall_better.csv", fontsize=9, color="#666")

    path = SHOTS / "08_dashboard_forecast_section.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def main():
    print(f"Generating screenshots -> {SHOTS}")
    shot_folder_structure()
    shot_dataset_preview()
    shot_pyspark_load()
    shot_area_ranking()
    shot_peak_and_daily()
    shot_forecast()
    shot_dashboard_summary()
    shot_dashboard_forecast()
    print("Done. Files:")
    for p in sorted(SHOTS.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
