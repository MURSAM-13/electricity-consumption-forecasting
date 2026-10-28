# Electricity Consumption Analysis & Forecasting

**B.Tech Course Based Project — Big Data Computing Technologies**

End-to-end Big Data style pipeline that analyzes and forecasts electricity
consumption across multiple service areas using **Apache Spark (PySpark)** for
distributed processing, and regular Python for forecasting, charts, and an
interactive **Streamlit** dashboard.

```
Spark (distributed ETL + SQL analytics)
        ↓ aggregated series
pandas / scikit-learn / statsmodels (forecasting)
        ↓
matplotlib · seaborn · plotly · Streamlit (presentation)
```

---

## Project structure

```
electricity_forecasting/
├── data/
│   └── electricity_consumption.csv   # synthetic multi-area dataset (+ dirty rows)
├── src/
│   ├── spark_session.py              # SparkSession (local / YARN / Standalone)
│   ├── data_ingestion.py             # CSV read from local FS or HDFS
│   ├── preprocessing.py              # distributed cleaning + date features
│   ├── analysis.py                   # groupBy, rank, stats, Spark SQL, HiveQL notes
│   ├── forecasting.py                # trend + ARIMA/linear regression (7-day)
│   ├── visualization.py              # PNG + optional Plotly HTML charts
│   └── dashboard.py                  # Streamlit interactive UI
├── output/                           # CSV exports + charts (created at runtime)
├── requirements.txt
├── README.md
└── main.py                           # full pipeline orchestrator
```

---

## Dataset

| Column            | Description                                      |
|-------------------|--------------------------------------------------|
| `date`            | Calendar day (`YYYY-MM-DD`)                      |
| `area`            | Service area (5 zones)                           |
| `consumption_kwh` | Electricity used (kWh)                           |
| `temperature_c`   | Daily temperature (°C)                           |
| `hour`            | Optional hour stamp (may be empty)               |

- **5 areas**: North / South / East / West / Central Zone  
- **90 days** of synthetic but realistic demand (seasonality, weekends, noise)  
- **Intentionally dirty rows**: missing values, `NA`, bad types, bad dates,
  extreme outliers, messy area names — cleaned in `preprocessing.py`

---

## Hadoop / Big Data concepts covered

| Concept | Where it appears in this project |
|--------|-----------------------------------|
| **HDFS storage** | Same Spark `read.csv` API accepts `hdfs://namenode:port/...` (see below) |
| **Distributed processing (MapReduce-like)** | Spark DataFrame ops partition data across cores/executors; groupBy/agg use shuffle stages like Map → Reduce |
| **Pig-style ETL** | `preprocessing.py` chains lazy transforms (trim, cast, filter, feature eng) without hand-written mappers |
| **Hive / SQL-on-Hadoop** | `analysis.py` registers a temp view and runs **Spark SQL**; README + code include **HiveQL** equivalents |
| **Caching** | Cleaned frame is `.cache()`'d so multiple actions reuse lineage |
| **Separation of concerns** | Heavy aggregation on Spark; compact time series forecast on the driver (common production pattern) |

---

## Prerequisites

1. **Python 3.9+**
2. **JDK 8, 11, or 17** with `JAVA_HOME` set (required by PySpark)
3. (Optional) Hadoop HDFS + YARN for cluster / HDFS demos

```bash
# Windows (example)
set JAVA_HOME=C:\Program Files\Java\jdk-17

# Linux / macOS
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
```

---

## Setup (local mode)

```bash
cd electricity_forecasting

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the full pipeline

```bash
python main.py
```

Artifacts land in `output/`:

- `daily_totals.csv`, `area_ranking.csv`, `peak_day.csv`, `summary_stats.csv`, `area_daily.csv`
- `forecast_overall_trend.csv`, `forecast_overall_better.csv`, `forecast_per_area.csv`
- `chart_daily_trend.png`, `chart_area_bar.png`, `chart_forecast.png`, `chart_distribution.png`, …
- `chart_forecast_interactive.html` (Plotly)

### Launch the dashboard

After at least one successful pipeline run:

```bash
# Option A — pipeline then UI
python main.py --dashboard

# Option B — UI only
streamlit run src/dashboard.py
# or
python main.py --dashboard-only
```

Filter by **area**, switch forecast method, and inspect summary cards, trends,
and forecast tables interactively.

---

## Switching the input path to HDFS

The ingestion code is path-agnostic. Spark treats `hdfs://` URIs the same way
as local paths — NameNode supplies block locations; executors read splits in
parallel (the Map side of classic MapReduce).

### 1. Put the file on HDFS

```bash
hdfs dfs -mkdir -p /user/data
hdfs dfs -put data/electricity_consumption.csv /user/data/
hdfs dfs -ls /user/data
```

### 2. Point the app at HDFS (pick one)

```bash
# Environment variable (no code change)
set ELECTRICITY_DATA_PATH=hdfs://localhost:9000/user/data/electricity_consumption.csv
python main.py

# CLI flag using the built-in example URI
python main.py --hdfs

# Explicit path
python main.py --data hdfs://<namenode>:9000/user/data/electricity_consumption.csv
```

### 3. Run on a cluster (YARN example)

```bash
set SPARK_MASTER=yarn
# or
python main.py --master yarn

# Production-style submit:
spark-submit --master yarn --deploy-mode client main.py
```

See also:

```bash
python main.py --show-hdfs-help
```

---

## Module map

| Module | Role | Big Data analogy |
|--------|------|------------------|
| `spark_session.py` | Create `SparkSession` | Job driver / cluster entry |
| `data_ingestion.py` | Read CSV (local / HDFS) | HDFS InputFormat / FileInputFormat |
| `preprocessing.py` | Clean, cast, features, cache | Pig Latin / MR mapper chain |
| `analysis.py` | Aggregations, rank, Spark SQL, HiveQL notes | HiveQL / Pig GROUP / reducers |
| `forecasting.py` | 7-day trend + ARIMA/linear | Downstream model service |
| `visualization.py` | Static + interactive charts | BI / reporting layer |
| `dashboard.py` | Streamlit UI with area filter | Ops dashboard |
| `main.py` | Orchestrates the pipeline | Workflow driver (Oozie-like) |

---

## Forecasting methods

1. **Simple trend** — last value + \(k \times\) mean daily change  
2. **Better model** — **ARIMA** via `statsmodels` when the series is long enough;
   automatic fallback to **scikit-learn Linear Regression** on the day index  

Forecasts are produced for:

- overall system total (next 7 days)
- each area independently (linear regression)

---

## Sample CLI

```bash
python main.py --horizon 7
python main.py --data data/electricity_consumption.csv --dashboard
python main.py --master local[2]
python main.py --show-hdfs-help
```

---

## HiveQL bonus (reference)

The same aggregations expressed for Hive / Beeline are printed by `analysis.py`
and included here for viva / documentation:

```sql
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
```

In Spark these run as `spark.sql(...)` on a temporary view — same declarative
SQL surface, distributed execution engine underneath.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Java gateway process exited` | Install JDK and set `JAVA_HOME` |
| `winutils` / Hadoop native warnings on Windows | Harmless for `local[*]`; or install winutils for Hadoop 3.x |
| Streamlit blank / missing data | Run `python main.py` once so `output/*.csv` exist |
| ARIMA convergence warnings | Expected on short series; code falls back to linear regression |
| Very slow first run | Spark JVM startup + package imports; later runs are faster |

---

## License / academic use

Prepared as a clean, modular **Course Based Project** template for Big Data
Computing Technologies. Synthetic dataset only — not real utility data.
