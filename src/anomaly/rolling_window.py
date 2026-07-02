"""Rolling-window statistical anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore_anomalies(
    series: pd.Series,
    window: int = 30,
    threshold: float = 3.0,
) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return (z_scores.abs() > threshold).astype(int)


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 30,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag timesteps where any sensor exceeds rolling z-score threshold."""
    result = df.copy()
    flags = pd.DataFrame(index=df.index)

    for col in sensor_cols:
        flags[f"{col}_anomaly"] = rolling_zscore_anomalies(df[col], window, threshold)

    result["rolling_anomaly"] = flags.max(axis=1)
    return result
