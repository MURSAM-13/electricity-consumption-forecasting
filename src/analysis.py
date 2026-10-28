"""
analysis.py
-----------
Big Data concept: Distributed aggregation & Spark SQL (Hive / Pig replacement).

Operations here (groupBy, agg, window rank, summary stats) are the Spark
equivalent of:
  - HiveQL  GROUP BY / ORDER BY / analytic functions
  - Pig     GROUP / FOREACH / nested bags
  - MapReduce Combiner + Reducer for sums/counts

Results are also exported to CSV under output/ for reporting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def _output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def daily_total_consumption(df: DataFrame) -> DataFrame:
    """
    Daily total consumption across all areas.
    Spark: groupBy + sum  ~  MapReduce sum combiner / Hive GROUP BY date
    """
    daily = (
        df.groupBy("date")
        .agg(
            F.sum("consumption_kwh").alias("total_kwh"),
            F.avg("consumption_kwh").alias("avg_kwh"),
            F.avg("temperature_c").alias("avg_temp_c"),
            F.count("*").alias("records"),
        )
        .orderBy("date")
    )
    return daily


def area_wise_consumption_and_rank(df: DataFrame) -> DataFrame:
    """
    Area-wise totals + rank (1 = highest consumer).
    Uses a Window function - same idea as Hive RANK() OVER (...).
    """
    area_totals = (
        df.groupBy("area")
        .agg(
            F.sum("consumption_kwh").alias("total_kwh"),
            F.avg("consumption_kwh").alias("avg_kwh"),
            F.stddev("consumption_kwh").alias("stddev_kwh"),
            F.count("*").alias("days_or_records"),
        )
    )
    w = Window.orderBy(F.col("total_kwh").desc())
    ranked = area_totals.withColumn("rank", F.rank().over(w))
    return ranked.orderBy("rank")


def peak_consumption_day(df: DataFrame) -> DataFrame:
    """Day with the highest total system consumption."""
    daily = daily_total_consumption(df)
    peak = daily.orderBy(F.col("total_kwh").desc()).limit(1)
    return peak


def summary_statistics(df: DataFrame) -> DataFrame:
    """
    Mean, stddev, min, max, variance of consumption_kwh.
    Spark's DataFrame.agg maps to distributed partial aggregates.
    """
    stats = df.agg(
        F.count("consumption_kwh").alias("count"),
        F.mean("consumption_kwh").alias("mean"),
        F.stddev("consumption_kwh").alias("stddev"),
        F.min("consumption_kwh").alias("min"),
        F.max("consumption_kwh").alias("max"),
        F.variance("consumption_kwh").alias("variance"),
        F.expr("percentile_approx(consumption_kwh, 0.5)").alias("median"),
    )
    return stats


def area_daily_series(df: DataFrame) -> DataFrame:
    """Per-area daily totals (used by forecasting & dashboard)."""
    return (
        df.groupBy("date", "area")
        .agg(F.sum("consumption_kwh").alias("total_kwh"),
             F.avg("temperature_c").alias("avg_temp_c"))
        .orderBy("area", "date")
    )


def run_spark_sql_equivalents(spark: SparkSession, df: DataFrame) -> dict[str, DataFrame]:
    """
    Register a temp view and re-express key analyses in Spark SQL.

    Big Data concept: Spark SQL ~ HiveQL on the same distributed engine.
    Temp views are session-scoped (like Hive temporary tables).
    """
    df.createOrReplaceTempView("electricity")

    sql_daily = spark.sql(
        """
        SELECT date,
               SUM(consumption_kwh) AS total_kwh,
               AVG(consumption_kwh) AS avg_kwh,
               AVG(temperature_c)   AS avg_temp_c,
               COUNT(*)             AS records
        FROM electricity
        GROUP BY date
        ORDER BY date
        """
    )

    sql_area_rank = spark.sql(
        """
        SELECT area,
               SUM(consumption_kwh) AS total_kwh,
               AVG(consumption_kwh) AS avg_kwh,
               RANK() OVER (ORDER BY SUM(consumption_kwh) DESC) AS rank
        FROM electricity
        GROUP BY area
        ORDER BY rank
        """
    )

    sql_peak = spark.sql(
        """
        SELECT date, SUM(consumption_kwh) AS total_kwh
        FROM electricity
        GROUP BY date
        ORDER BY total_kwh DESC
        LIMIT 1
        """
    )

    sql_stats = spark.sql(
        """
        SELECT COUNT(consumption_kwh) AS count,
               AVG(consumption_kwh)   AS mean,
               STDDEV(consumption_kwh) AS stddev,
               MIN(consumption_kwh)   AS min,
               MAX(consumption_kwh)   AS max,
               VARIANCE(consumption_kwh) AS variance
        FROM electricity
        """
    )

    return {
        "daily": sql_daily,
        "area_rank": sql_area_rank,
        "peak": sql_peak,
        "stats": sql_stats,
    }


def hiveql_reference() -> str:
    """
    Bonus: equivalent HiveQL a student could run in the Hive CLI / Beeline
    against an external table over the same CSV on HDFS.
    """
    return """
    -- ========== HiveQL equivalents (run in Beeline / Hive CLI) ==========
    -- Assume:
    --   CREATE EXTERNAL TABLE electricity (
    --     date STRING, area STRING, consumption_kwh DOUBLE,
    --     temperature_c DOUBLE, hour INT
    --   )
    --   ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
    --   STORED AS TEXTFILE
    --   LOCATION 'hdfs:///user/data/electricity'
    --   TBLPROPERTIES ('skip.header.line.count'='1');

    -- Daily totals
    SELECT date, SUM(consumption_kwh) AS total_kwh
    FROM electricity
    GROUP BY date
    ORDER BY date;

    -- Area ranking
    SELECT area,
           SUM(consumption_kwh) AS total_kwh,
           RANK() OVER (ORDER BY SUM(consumption_kwh) DESC) AS rank
    FROM electricity
    GROUP BY area;

    -- Peak day
    SELECT date, SUM(consumption_kwh) AS total_kwh
    FROM electricity
    GROUP BY date
    ORDER BY total_kwh DESC
    LIMIT 1;

    -- Summary stats
    SELECT AVG(consumption_kwh), STDDEV(consumption_kwh),
           MIN(consumption_kwh), MAX(consumption_kwh),
           VARIANCE(consumption_kwh)
    FROM electricity;
    -- ====================================================================
    """


def export_analysis_csvs(
    daily: DataFrame,
    area_rank: DataFrame,
    peak: DataFrame,
    stats: DataFrame,
    area_daily: DataFrame,
) -> dict[str, str]:
    """
    Write analysis results as single CSV files under output/.

    Course-scale results are collected to the driver and written with pandas.
    That avoids the Windows winutils requirement of Spark's Hadoop file sink
    while still computing the aggregations distributively in Spark.

    On a real cluster / HDFS you would instead use::

        sdf.coalesce(1).write.mode("overwrite").option("header", "true").csv(
            "hdfs://namenode:9000/user/output/daily_totals"
        )
    """
    out = _output_dir()
    paths: dict[str, str] = {}

    def _write_single(sdf: DataFrame, name: str) -> str:
        # Distributed compute already done; collect small result set to driver
        pdf = sdf.toPandas()
        single = out / f"{name}.csv"
        pdf.to_csv(single, index=False)
        paths[name] = str(single)
        print(f"[Analysis] Exported {name} -> {single}")
        return str(single)

    _write_single(daily, "daily_totals")
    _write_single(area_rank, "area_ranking")
    _write_single(peak, "peak_day")
    _write_single(stats, "summary_stats")
    _write_single(area_daily, "area_daily")
    return paths


def run_full_analysis(spark: SparkSession, df: DataFrame) -> dict[str, Any]:
    """
    Orchestrate all analyses, print key results, run SQL twins, export CSVs.
    """
    print("\n" + "=" * 70)
    print(" ANALYSIS (PySpark distributed aggregations ~ Hive / Pig / MR)")
    print("=" * 70)

    daily = daily_total_consumption(df)
    area_rank = area_wise_consumption_and_rank(df)
    peak = peak_consumption_day(df)
    stats = summary_statistics(df)
    area_daily = area_daily_series(df)

    print("\n--- Area ranking (1 = highest total consumption) ---")
    area_rank.show(truncate=False)

    print("--- Peak consumption day ---")
    peak.show(truncate=False)

    print("--- Summary statistics (consumption_kwh) ---")
    stats.show(truncate=False)

    print("--- Daily totals (sample first 5) ---")
    daily.show(5, truncate=False)

    print("\n--- Spark SQL equivalents (Hive-like) ---")
    sql_results = run_spark_sql_equivalents(spark, df)
    print("SQL area rank:")
    sql_results["area_rank"].show(truncate=False)
    print("SQL peak day:")
    sql_results["peak"].show(truncate=False)

    print(hiveql_reference())

    paths = export_analysis_csvs(daily, area_rank, peak, stats, area_daily)

    # Collect small results for main() printing / forecasting hand-off
    results = {
        "daily": daily,
        "area_rank": area_rank,
        "peak": peak,
        "stats": stats,
        "area_daily": area_daily,
        "sql": sql_results,
        "export_paths": paths,
        "area_rank_rows": [r.asDict() for r in area_rank.collect()],
        "peak_row": peak.collect()[0].asDict() if peak.count() else {},
        "stats_row": stats.collect()[0].asDict() if stats.count() else {},
    }
    return results
