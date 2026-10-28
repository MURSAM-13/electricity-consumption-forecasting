#!/usr/bin/env python3
"""
main.py
-------
End-to-end pipeline orchestrator for
  "Electricity Consumption Analysis & Forecasting"

Flow (Big Data storyline):
  1. SparkSession          -> cluster/local entry (YARN/Standalone/local[*])
  2. data_ingestion        -> read CSV from local FS or HDFS
  3. preprocessing         -> distributed ETL + cache
  4. analysis              -> groupBy / window / Spark SQL (Hive-like)
  5. forecasting           -> collect aggregates -> pandas / sklearn / ARIMA
  6. visualization         -> matplotlib / seaborn / plotly charts -> output/
  7. (optional) dashboard  -> Streamlit UI over exported CSVs

Usage
-----
  python main.py                     # full pipeline
  python main.py --dashboard         # pipeline then open Streamlit
  python main.py --dashboard-only    # only Streamlit (needs prior run)
  python main.py --hdfs              # read from example HDFS URI
  python main.py --data path/to.csv  # custom input path

Environment
-----------
  SPARK_MASTER=local[*]|yarn|spark://...
  ELECTRICITY_DATA_PATH=hdfs://namenode:9000/user/data/electricity_consumption.csv
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Make `src` importable when running `python main.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spark_session import create_spark_session, stop_spark_session
from src.data_ingestion import ingest_electricity_data, show_hdfs_usage_notes
from src.preprocessing import preprocess_electricity_data
from src.analysis import run_full_analysis
from src.forecasting import run_forecasting
from src.visualization import generate_all_visualizations
from src.dashboard import launch_dashboard, build_dashboard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Electricity Consumption Analysis & Forecasting (PySpark)",
    )
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV (local path or hdfs://...). Overrides defaults.",
    )
    p.add_argument(
        "--hdfs",
        action="store_true",
        help="Use the example HDFS URI (see data_ingestion.EXAMPLE_HDFS_CSV).",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        help="After the pipeline, launch the Streamlit dashboard.",
    )
    p.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Skip Spark pipeline; only launch Streamlit (requires output/ CSVs).",
    )
    p.add_argument(
        "--master",
        type=str,
        default=None,
        help='Spark master URL (default: env SPARK_MASTER or "local[*]").',
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Forecast horizon in days (default: 7).",
    )
    p.add_argument(
        "--show-hdfs-help",
        action="store_true",
        help="Print HDFS setup notes and exit.",
    )
    return p.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    spark = None
    try:
        print("\n" + "#" * 70)
        print("#  ELECTRICITY CONSUMPTION ANALYSIS & FORECASTING")
        print("#  Big Data Computing Technologies - Course Based Project")
        print("#" * 70)

        # 1. SparkSession --------------------------------------------------------
        spark = create_spark_session(
            app_name="ElectricityConsumptionAnalysis",
            master=args.master,
        )

        # 2. Ingest --------------------------------------------------------------
        raw_df = ingest_electricity_data(
            spark,
            path=args.data,
            use_hdfs=args.hdfs,
        )
        print("\n[Raw sample]")
        raw_df.show(5, truncate=False)

        # 3. Preprocess ----------------------------------------------------------
        clean_df = preprocess_electricity_data(raw_df, cache=True)
        print("\n[Clean sample]")
        clean_df.show(5, truncate=False)

        # 4. Analyze -------------------------------------------------------------
        analysis = run_full_analysis(spark, clean_df)

        # 5. Forecast ------------------------------------------------------------
        forecast = run_forecasting(
            analysis["daily"],
            analysis["area_daily"],
            horizon=args.horizon,
        )

        # 6. Visualize -----------------------------------------------------------
        # Small sample of clean rows for distribution plots (driver-side)
        clean_sample = clean_df.select(
            "date", "area", "consumption_kwh", "temperature_c"
        ).toPandas()

        area_rank_pdf = analysis["area_rank"].toPandas()
        viz_paths = generate_all_visualizations(
            daily_pdf=forecast["daily_history"],
            area_rank_pdf=area_rank_pdf,
            forecast_better=forecast["forecast_better"],
            area_forecast=forecast["forecast_area"],
            clean_sample_pdf=clean_sample,
        )

        # 7. Key results summary -------------------------------------------------
        print("\n" + "=" * 70)
        print(" KEY RESULTS SUMMARY")
        print("=" * 70)

        print("\nArea ranking:")
        for row in analysis["area_rank_rows"]:
            print(
                f"  #{row.get('rank', '?')}  {row.get('area', '?'):15s}  "
                f"total={row.get('total_kwh', 0):,.2f} kWh  "
                f"avg={row.get('avg_kwh', 0):,.2f} kWh"
            )

        peak = analysis["peak_row"]
        if peak:
            print(
                f"\nPeak day: {peak.get('date')}  "
                f"total={peak.get('total_kwh', 0):,.2f} kWh"
            )

        stats = analysis["stats_row"]
        if stats:
            print(
                f"\nConsumption stats - mean={stats.get('mean', 0):,.2f}  "
                f"std={stats.get('stddev', 0):,.2f}  "
                f"min={stats.get('min', 0):,.2f}  max={stats.get('max', 0):,.2f}"
            )

        print("\n7-day overall forecast (better model):")
        for _, r in forecast["forecast_better"].iterrows():
            print(f"  {r['date'].date()}  ->  {r['forecast_kwh']:,.2f} kWh  [{r['method']}]")

        print("\nArtifacts written under:", PROJECT_ROOT / "output")
        for label, path in {**analysis["export_paths"], **forecast["paths"]}.items():
            print(f"  - {label}: {path}")
        for label, path in viz_paths.items():
            if path:
                print(f"  - chart/{label}: {path}")

        print("\nPipeline completed successfully.")
        return 0

    except Exception as exc:  # noqa: BLE001
        print("\n[ERROR] Pipeline failed:", exc, file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if spark is not None:
            stop_spark_session(spark)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.show_hdfs_help:
        show_hdfs_usage_notes()
        return 0

    if args.dashboard_only:
        # When imported under streamlit this branch is not used; CLI helper only.
        launch_dashboard()
        return 0

    code = run_pipeline(args)

    if code == 0 and args.dashboard:
        launch_dashboard()

    return code


if __name__ == "__main__":
    raise SystemExit(main())
