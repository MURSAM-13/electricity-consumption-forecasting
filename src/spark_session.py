"""
spark_session.py
----------------
Big Data concept: Cluster resource manager / Spark driver entry point.

Creates a SparkSession - the unified entry point to Spark SQL, DataFrames,
and the underlying distributed execution engine (similar in role to a
MapReduce JobClient + HiveServer session combined).

Local mode  : all executors run inside one JVM on this machine (dev/demo).
Cluster mode: connect to YARN / Standalone / Kubernetes and read from HDFS.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

from pyspark.sql import SparkSession


def _ensure_compatible_java() -> None:
    """
    PySpark 3.4/3.5 is tested with JDK 8/11/17. Newer JDKs (21+) may raise
    `getSubject is supported only if a security manager is allowed`.

    If JAVA_HOME looks incompatible, try to auto-select JDK 8/11/17 on Windows.
    Users can still override by exporting JAVA_HOME themselves.
    """
    current = os.environ.get("JAVA_HOME", "")
    ver_hint = current.lower()

    def _looks_ok(path: str) -> bool:
        p = path.lower().replace("\\", "/")
        # Accept 1.8, 8, 11, 17 explicitly; reject 21+
        bad = any(x in p for x in ("jdk-2", "jdk-21", "jdk-22", "jdk-23", "jdk-24", "jdk-25"))
        if bad:
            return False
        return bool(path) and Path(path, "bin").exists()

    if _looks_ok(current):
        return

    candidates: list[str] = []
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    patterns = [
        str(Path(program_files) / "Java" / "jdk1.8*"),
        str(Path(program_files) / "Java" / "jdk-17*"),
        str(Path(program_files) / "Java" / "jdk-11*"),
        str(Path(program_files) / "Java" / "jdk-8*"),
        str(Path(program_files) / "Eclipse Adoptium" / "jdk-17*"),
        str(Path(program_files) / "Eclipse Adoptium" / "jdk-11*"),
        str(Path(program_files) / "Eclipse Adoptium" / "jdk-8*"),
        str(Path(program_files) / "Microsoft" / "jdk-17*"),
        str(Path(program_files) / "Microsoft" / "jdk-11*"),
    ]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat), reverse=True))

    for cand in candidates:
        if _looks_ok(cand):
            os.environ["JAVA_HOME"] = cand
            # Prepend Java bin so child JVMs resolve correctly
            bin_dir = str(Path(cand) / "bin")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"[SparkSession] Auto-selected compatible JAVA_HOME={cand}")
            return

    if current:
        print(
            f"[SparkSession] WARNING: JAVA_HOME={current} may be too new for PySpark. "
            "Prefer JDK 8, 11, or 17 if Spark fails to start."
        )


def create_spark_session(
    app_name: str = "ElectricityConsumptionAnalysis",
    master: str | None = None,
    enable_hive: bool = False,
) -> SparkSession:
    """
    Build and return a SparkSession.

    Parameters
    ----------
    app_name : str
        Application name shown in the Spark UI / YARN ResourceManager.
    master : str | None
        Spark master URL. If None, uses SPARK_MASTER env var or "local[*]".
        Examples:
          - "local[*]"              -> use all local CPU cores (default)
          - "local[2]"              -> 2 local threads
          - "yarn"                  -> submit to a Hadoop YARN cluster
          - "spark://host:7077"     -> Spark Standalone cluster
    enable_hive : bool
        If True, enable Hive support (metastore / HiveQL compatibility).
        Requires a configured hive-site.xml on a real cluster.

    How to switch to HDFS / cluster mode
    ------------------------------------
    1. Package the job and submit with spark-submit:
         spark-submit --master yarn --deploy-mode cluster main.py
    2. Or set environment variables before running:
         set SPARK_MASTER=yarn
         set HADOOP_CONF_DIR=C:\\hadoop\\etc\\hadoop
    3. Point data paths to hdfs://namenode:9000/user/... (see data_ingestion.py).

    Returns
    -------
    SparkSession
    """
    _ensure_compatible_java()

    if master is None:
        master = os.environ.get("SPARK_MASTER", "local[*]")

    # Quiet Hadoop native / winutils noise on Windows local mode
    os.environ.setdefault("HADOOP_HOME", os.environ.get("HADOOP_HOME", os.path.abspath(".")))

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        # Reasonable defaults for a course/demo workload
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # Disable UI in constrained environments if needed:
        # .config("spark.ui.enabled", "false")
    )

    if enable_hive:
        # Demonstrates Hive integration (metastore + HiveQL)
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[SparkSession] app='{app_name}' master='{master}' "
          f"spark.version={spark.version}")
    return spark


def stop_spark_session(spark: SparkSession) -> None:
    """Gracefully stop the session and free cluster/local resources."""
    if spark is not None:
        spark.stop()
        print("[SparkSession] stopped.")
