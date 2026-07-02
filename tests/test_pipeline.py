"""Tests for predictive maintenance pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_synthetic_data import generate_machine_series, generate_dataset
from src.data.preprocessing import create_sequences, prepare_failure_prediction_data, SENSOR_COLS
from src.anomaly.rolling_window import rolling_zscore_anomalies, detect_multivariate_anomalies
from src.models.sequence_models import build_lstm_model, build_gru_model
from src.models.autoencoder import build_autoencoder, fit_threshold, reconstruction_errors


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_generate_machine_series_shape(rng):
    df = generate_machine_series(0, 500, rng, failure_rate=0.1)
    assert len(df) == 500
    assert set(SENSOR_COLS).issubset(df.columns)
    assert df["failure"].sum() > 0


def test_generate_dataset(rng):
    config = {
        "project": {"random_seed": 42},
        "data": {
            "n_machines": 3,
            "samples_per_machine": 100,
            "failure_rate": 0.1,
        },
    }
    df = generate_dataset(config)
    assert len(df) == 300
    assert df["machine_id"].nunique() == 3


def test_create_sequences():
    features = np.random.randn(100, 3)
    targets = np.zeros(100)
    targets[-10:] = 1
    X, y = create_sequences(features, targets, window_size=10, horizon=5)
    assert X.shape == (86, 10, 3)
    assert len(y) == 86


def test_prepare_failure_prediction_data():
    rng = np.random.default_rng(0)
    frames = [generate_machine_series(i, 200, rng, 0.1) for i in range(5)]
    df = pd.concat(frames, ignore_index=True)
    data = prepare_failure_prediction_data(df, window_size=20, horizon=5)
    assert data["X_train"].ndim == 3
    assert len(data["y_test"]) == len(data["X_test"])


def test_rolling_zscore_anomalies():
    series = pd.Series([1.0] * 50 + [100.0])
    flags = rolling_zscore_anomalies(series, window=10, threshold=2.0)
    assert flags.iloc[-1] == 1


def test_detect_multivariate_anomalies(rng):
    df = generate_machine_series(0, 100, rng, 0.05)
    result = detect_multivariate_anomalies(df, SENSOR_COLS, window=10, threshold=2.0)
    assert "rolling_anomaly" in result.columns


def test_lstm_model_build():
    model = build_lstm_model((50, 3))
    assert model.output_shape == (None, 1)


def test_gru_model_build():
    model = build_gru_model((50, 3))
    assert model.output_shape == (None, 1)


def test_autoencoder_threshold():
    errors = np.array([0.1, 0.2, 0.15, 0.9, 1.0])
    threshold = fit_threshold(errors, percentile=80)
    assert threshold >= 0.15


def test_autoencoder_reconstruction_errors():
    model = build_autoencoder(input_dim=3, encoding_dim=2)
    X = np.random.randn(20, 3).astype(np.float32)
    errors = reconstruction_errors(model, X)
    assert errors.shape == (20,)
