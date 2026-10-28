"""
data_ingestion.py
-----------------
Big Data concept: HDFS / distributed file system ingestion.

In production, large CSVs (or Parquet/ORC) live on HDFS. Spark reads them
in parallel - each executor reads a split (block), the same idea as the
Map phase of MapReduce reading HDFS InputSplits.

The SAME Spark read API works for:
  - local filesystem : file:///... or plain relative path
  - HDFS             : hdfs://<namenode>:<port>/path/to/file.csv
  - S3 / ADLS / GCS  : s3a://...  (with appropriate connectors)
"""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# Default local dataset relative to project root
DEFAULT_LOCAL_CSV = os.path.join("data", "electricity_consumption.csv")

# Example HDFS path (change namenode host/port for your cluster)
EXAMPLE_HDFS_CSV = "hdfs://localhost:9000/user/data/electricity_consumption.csv"


def resolve_data_path(path: str | None = None, use_hdfs: bool = False) -> str:
    """
    Resolve the input path.

    Priority:
      1. Explicit `path` argument
      2. Env var ELECTRICITY_DATA_PATH
      3. HDFS example path if use_hdfs=True
      4. Local default CSV under project data/

    To use HDFS without code changes:
      set ELECTRICITY_DATA_PATH=hdfs://namenode:9000/user/data/electricity_consumption.csv
    """
    if path:
        return path

    env_path = os.environ.get("ELECTRICITY_DATA_PATH")
    if env_path:
        return env_path

    if use_hdfs:
        print(f"[Ingestion] use_hdfs=True -> {EXAMPLE_HDFS_CSV}")
        print("            Ensure the file was put with: "
              "hdfs dfs -put data/electricity_consumption.csv /user/data/")
        return EXAMPLE_HDFS_CSV

    # Resolve relative to project root (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    local = project_root / DEFAULT_LOCAL_CSV
    return str(local)


def ingest_electricity_data(
    spark: SparkSession,
    path: str | None = None,
    use_hdfs: bool = False,
) -> DataFrame:
    """
    Read the electricity consumption CSV into a Spark DataFrame.

    Demonstrates:
      - Distributed text/CSV input (MapReduce InputFormat analogue)
      - Schema inference + later explicit casting in preprocessing
      - Header handling (header=True so first row is not treated as data)

    Parameters
    ----------
    spark : SparkSession
    path : optional override path (local or hdfs://...)
    use_hdfs : if True and path is None, use EXAMPLE_HDFS_CSV

    Returns
    -------
    DataFrame with raw columns as strings/inferred types (still dirty).
    """
    data_path = resolve_data_path(path=path, use_hdfs=use_hdfs)
    print(f"[Ingestion] Reading: {data_path}")

    # Spark parallel CSV reader - works for local AND HDFS paths.
    # multiLine/escape options omitted; data is simple single-line CSV.
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")   # keep as string; cast in preprocessing
        .option("mode", "PERMISSIVE")     # do not fail hard on malformed rows
        .option("nullValue", "")
        .option("nanValue", "NA")
        .csv(data_path)
    )

    # Normalize column names (trim / lower) - defensive against dirty headers
    for col_name in df.columns:
        clean = col_name.strip().lower()
        if clean != col_name:
            df = df.withColumnRenamed(col_name, clean)

    row_count = df.count()
    print(f"[Ingestion] Loaded {row_count} raw rows, columns={df.columns}")
    df.printSchema()
    return df


def show_hdfs_usage_notes() -> None:
    """Print how the identical Spark code targets HDFS (course documentation)."""
    notes = """
    -------------------------------------------------------------------------
    HDFS usage (same Spark code, different URI)
    -------------------------------------------------------------------------
    1. Start HDFS (pseudo-distributed):
         start-dfs.sh

    2. Create a directory and upload the CSV:
         hdfs dfs -mkdir -p /user/data
         hdfs dfs -put data/electricity_consumption.csv /user/data/

    3. Run this project with:
         set ELECTRICITY_DATA_PATH=hdfs://localhost:9000/user/data/electricity_consumption.csv
         python main.py

       OR call:
         ingest_electricity_data(spark, use_hdfs=True)

    Spark will ask the NameNode for block locations and assign one task per
    HDFS split - this is the distributed "map" side of the pipeline.
    -------------------------------------------------------------------------
    """
    print(notes)
