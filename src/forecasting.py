"""
forecasting.py
--------------
Driver-side forecasting on aggregated series.

Big Data split of responsibility:
  - Heavy history aggregation stays in Spark (cluster / local[*]).
  - Compact daily time series is collected to the driver as pandas, then
    classical ML / statsmodels run in regular Python (not distributed).

This mirrors real architectures: Spark for ETL + feature marts, then a
single-node model service for short-horizon forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

try:
    from sklearn.linear_model import LinearRegression
except ImportError:  # pragma: no cover
    LinearRegression = None  # type: ignore

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover
    ARIMA = None  # type: ignore


def _output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def spark_daily_to_pandas(daily_df: DataFrame) -> pd.DataFrame:
    """Collect Spark daily totals to a sorted pandas DataFrame."""
    pdf = daily_df.select("date", "total_kwh").toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("date").reset_index(drop=True)
    return pdf


def spark_area_daily_to_pandas(area_daily_df: DataFrame) -> pd.DataFrame:
    pdf = area_daily_df.select("date", "area", "total_kwh").toPandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values(["area", "date"]).reset_index(drop=True)
    return pdf


# ---------------------------------------------------------------------------
# Method 1 - Simple trend (average daily change)
# ---------------------------------------------------------------------------
def forecast_simple_trend(series: pd.Series, horizon: int = 7) -> np.ndarray:
    """
    Naïve linear trend: last_value + k * mean(diff(series)), k = 1..horizon.
    Fast baseline; no external deps beyond numpy/pandas.
    """
    s = series.astype(float).dropna()
    if len(s) < 2:
        last = float(s.iloc[-1]) if len(s) else 0.0
        return np.full(horizon, last)

    avg_change = float(s.diff().dropna().mean())
    last = float(s.iloc[-1])
    return np.array([last + (i + 1) * avg_change for i in range(horizon)], dtype=float)


# ---------------------------------------------------------------------------
# Method 2 - Linear Regression on time index (+ optional weekday dummies)
# ---------------------------------------------------------------------------
def forecast_linear_regression(series: pd.Series, horizon: int = 7) -> np.ndarray:
    """Ordinary least squares on day index (and day-of-week if dates available)."""
    s = series.astype(float).dropna()
    n = len(s)
    if n < 3 or LinearRegression is None:
        return forecast_simple_trend(s, horizon)

    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = s.values
    model = LinearRegression()
    model.fit(X, y)
    X_future = np.arange(n, n + horizon, dtype=float).reshape(-1, 1)
    preds = model.predict(X_future)
    # Consumption cannot be negative
    return np.maximum(preds, 0.0)


# ---------------------------------------------------------------------------
# Method 2b - ARIMA (statsmodels) when available
# ---------------------------------------------------------------------------
def forecast_arima(series: pd.Series, horizon: int = 7, order=(2, 1, 2)) -> np.ndarray:
    """
    ARIMA short-horizon forecast. Falls back to linear regression on failure
    (short series, non-convergence, missing statsmodels).
    """
    s = series.astype(float).dropna()
    if ARIMA is None or len(s) < 15:
        return forecast_linear_regression(s, horizon)

    try:
        model = ARIMA(s.values, order=order)
        fitted = model.fit()
        fc = fitted.forecast(steps=horizon)
        return np.maximum(np.asarray(fc, dtype=float), 0.0)
    except Exception as exc:  # noqa: BLE001 - course demo resilience
        print(f"[Forecast] ARIMA failed ({exc}); falling back to linear regression.")
        return forecast_linear_regression(s, horizon)


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


def forecast_overall(
    daily_pdf: pd.DataFrame,
    horizon: int = 7,
    method: str = "arima",
) -> pd.DataFrame:
    """
    Forecast next `horizon` days of system-wide total_kwh.

    method: 'trend' | 'linear' | 'arima'
    """
    series = daily_pdf.set_index("date")["total_kwh"]
    if method == "trend":
        values = forecast_simple_trend(series, horizon)
        method_used = "simple_trend"
    elif method == "linear":
        values = forecast_linear_regression(series, horizon)
        method_used = "linear_regression"
    else:
        values = forecast_arima(series, horizon)
        method_used = "arima" if ARIMA is not None else "linear_regression_fallback"

    dates = _future_dates(series.index.max(), horizon)
    out = pd.DataFrame(
        {
            "date": dates,
            "forecast_kwh": np.round(values, 2),
            "method": method_used,
            "scope": "overall",
        }
    )
    return out


def forecast_per_area(
    area_daily_pdf: pd.DataFrame,
    horizon: int = 7,
    method: str = "linear",
) -> pd.DataFrame:
    """Independent short forecast for each area."""
    frames = []
    for area, grp in area_daily_pdf.groupby("area"):
        series = grp.set_index("date")["total_kwh"].sort_index()
        if method == "trend":
            values = forecast_simple_trend(series, horizon)
            method_used = "simple_trend"
        elif method == "arima":
            values = forecast_arima(series, horizon)
            method_used = "arima" if ARIMA is not None else "linear_regression_fallback"
        else:
            values = forecast_linear_regression(series, horizon)
            method_used = "linear_regression"

        dates = _future_dates(series.index.max(), horizon)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "area": area,
                    "forecast_kwh": np.round(values, 2),
                    "method": method_used,
                    "scope": "area",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_forecasting(
    daily_spark: DataFrame,
    area_daily_spark: DataFrame,
    horizon: int = 7,
) -> dict[str, Any]:
    """
    End-to-end forecast pipeline called from main.py.
    Produces overall (trend + better model) and per-area forecasts.
    """
    print("\n" + "=" * 70)
    print(" FORECASTING (pandas / scikit-learn / statsmodels on driver)")
    print("=" * 70)

    daily_pdf = spark_daily_to_pandas(daily_spark)
    area_pdf = spark_area_daily_to_pandas(area_daily_spark)

    # Two methods as required
    fc_trend = forecast_overall(daily_pdf, horizon=horizon, method="trend")
    fc_better = forecast_overall(daily_pdf, horizon=horizon, method="arima")
    fc_area = forecast_per_area(area_pdf, horizon=horizon, method="linear")

    print("\n--- Overall forecast (simple trend) - next 7 days ---")
    print(fc_trend.to_string(index=False))

    print("\n--- Overall forecast (ARIMA / linear) - next 7 days ---")
    print(fc_better.to_string(index=False))

    print("\n--- Per-area forecast (linear regression) - sample ---")
    print(fc_area.head(15).to_string(index=False))

    out = _output_dir()
    p_trend = out / "forecast_overall_trend.csv"
    p_better = out / "forecast_overall_better.csv"
    p_area = out / "forecast_per_area.csv"
    p_history = out / "daily_history_for_forecast.csv"

    fc_trend.to_csv(p_trend, index=False)
    fc_better.to_csv(p_better, index=False)
    fc_area.to_csv(p_area, index=False)
    daily_pdf.to_csv(p_history, index=False)

    print(f"\n[Forecast] Saved:\n  {p_trend}\n  {p_better}\n  {p_area}")

    return {
        "daily_history": daily_pdf,
        "area_history": area_pdf,
        "forecast_trend": fc_trend,
        "forecast_better": fc_better,
        "forecast_area": fc_area,
        "paths": {
            "trend": str(p_trend),
            "better": str(p_better),
            "area": str(p_area),
            "history": str(p_history),
        },
    }
