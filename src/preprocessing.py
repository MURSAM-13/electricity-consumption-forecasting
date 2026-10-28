"""
preprocessing.py
----------------
Big Data concept: Distributed ETL / data cleansing (Pig / Spark map-side work).

All transformations below are lazy Spark DataFrame operations. They compile
to a DAG of stages executed across partitions - conceptually similar to a
Pig Latin script or a chain of MapReduce jobs, without writing mappers by hand.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType


# Canonical area names after cleaning
VALID_AREAS = [
    "North Zone",
    "South Zone",
    "East Zone",
    "West Zone",
    "Central Zone",
]


def preprocess_electricity_data(df: DataFrame, cache: bool = True) -> DataFrame:
    """
    Clean and enrich the raw electricity DataFrame.

    Steps (distributed):
      1. Trim / normalize string columns
      2. Cast types (string -> date, double, int)
      3. Drop rows with invalid dates, missing area, or non-numeric consumption
      4. Fill missing temperature with area-wise median (approx via approxQuantile)
      5. Remove extreme outliers (consumption > Q3 + 3*IQR style hard cap)
      6. Engineer date features: day, month, year, day_of_week, is_weekend
      7. Cache the result so later actions do not recompute the lineage

    Returns
    -------
    Cleaned, typed, feature-enriched DataFrame.
    """
    print("[Preprocessing] Starting distributed clean + feature engineering...")

    # --- 1. Basic string hygiene -------------------------------------------------
    df = df.select(
        F.trim(F.col("date")).alias("date"),
        F.trim(F.col("area")).alias("area"),
        F.trim(F.col("consumption_kwh")).alias("consumption_kwh"),
        F.trim(F.col("temperature_c")).alias("temperature_c"),
        F.trim(F.col("hour")).alias("hour"),
    )

    # Normalize area casing / spacing  (e.g. " south zone " -> "South Zone")
    df = df.withColumn(
        "area",
        F.initcap(F.lower(F.regexp_replace(F.col("area"), r"\s+", " "))),
    )

    # --- 2. Cast types -----------------------------------------------------------
    # Replace common dirty tokens before casting
    for dirty in ("NA", "N/A", "null", "None", "nan", "NaN"):
        df = df.replace(dirty, None, subset=["consumption_kwh", "temperature_c", "hour", "date"])

    df = (
        df.withColumn("consumption_kwh", F.col("consumption_kwh").cast(DoubleType()))
          .withColumn("temperature_c", F.col("temperature_c").cast(DoubleType()))
          .withColumn("hour", F.col("hour").cast(IntegerType()))
          .withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
    )

    # --- 3. Drop invalid core rows ----------------------------------------------
    before = df.count()
    df = df.filter(
        F.col("date").isNotNull()
        & F.col("area").isNotNull()
        & (F.length(F.col("area")) > 0)
        & F.col("consumption_kwh").isNotNull()
        & (F.col("consumption_kwh") > 0)
    )
    # Keep only known areas (after initcap normalization)
    df = df.filter(F.col("area").isin(VALID_AREAS))
    after_filter = df.count()
    print(f"[Preprocessing] Dropped {before - after_filter} dirty/invalid rows "
          f"({before} -> {after_filter})")

    # --- 4. Outlier removal (hard upper bound for course demo) ------------------
    # Synthetic dirty row used 999999 kWh; also drop absurd values.
    stats = df.select(
        F.expr("percentile_approx(consumption_kwh, 0.25)").alias("q1"),
        F.expr("percentile_approx(consumption_kwh, 0.75)").alias("q3"),
    ).collect()[0]
    q1, q3 = stats["q1"], stats["q3"]
    iqr = q3 - q1
    upper = q3 + 3.0 * iqr
    # Also absolute sanity cap
    upper = min(upper, 50_000.0)
    before_out = df.count()
    df = df.filter(F.col("consumption_kwh") <= F.lit(upper))
    print(f"[Preprocessing] Outlier filter upper={upper:.1f} kWh; "
          f"removed {before_out - df.count()} rows")

    # --- 5. Fill missing temperature (global median as simple distributed fill) -
    # approxQuantile is a distributed approximate algorithm (no full sort required)
    temp_median = df.approxQuantile("temperature_c", [0.5], 0.01)
    fill_temp = float(temp_median[0]) if temp_median and temp_median[0] is not None else 25.0
    df = df.fillna({"temperature_c": fill_temp})
    # hour is optional - leave nulls; downstream analysis does not require it
    print(f"[Preprocessing] Filled null temperature_c with median ~ {fill_temp:.1f}°C")

    # --- 6. Date feature engineering (distributed column expressions) -----------
    df = (
        df.withColumn("year", F.year("date"))
          .withColumn("month", F.month("date"))
          .withColumn("day", F.dayofmonth("date"))
          .withColumn("day_of_week", F.dayofweek("date"))  # 1=Sunday ... 7=Saturday
          .withColumn(
              "day_name",
              F.date_format("date", "EEEE"),
          )
          .withColumn(
              "is_weekend",
              F.when(F.dayofweek("date").isin(1, 7), F.lit(1)).otherwise(F.lit(0)),
          )
    )

    # Final column order
    df = df.select(
        "date",
        "area",
        "consumption_kwh",
        "temperature_c",
        "hour",
        "year",
        "month",
        "day",
        "day_of_week",
        "day_name",
        "is_weekend",
    )

    # --- 7. Cache ----------------------------------------------------------------
    # Big Data concept: MEMORY_AND_DISK cache avoids recomputing the ETL DAG
    # when multiple actions (count, groupBy, write) follow.
    if cache:
        df = df.cache()
        # Materialize once so cache is warm
        n = df.count()
        print(f"[Preprocessing] Cached cleaned DataFrame with {n} rows")
    else:
        print(f"[Preprocessing] Cleaned DataFrame ready (not cached)")

    df.printSchema()
    return df
