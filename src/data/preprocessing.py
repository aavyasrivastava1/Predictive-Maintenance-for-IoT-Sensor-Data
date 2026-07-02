"""Feature engineering and windowing for time-series models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import yaml


SENSOR_COLS = ["temperature", "vibration", "pressure"]


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(raw_dir: str | Path) -> pd.DataFrame:
    path = Path(raw_dir) / "iot_sensor_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run: python -m src.data.generate_synthetic_data"
        )
    return pd.read_csv(path, parse_dates=["timestamp"])


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - window_size - horizon + 1):
        X.append(features[i : i + window_size])
        y.append(targets[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


def prepare_failure_prediction_data(
    df: pd.DataFrame,
    window_size: int = 50,
    horizon: int = 10,
    test_split: float = 0.2,
    val_split: float = 0.1,
) -> dict:
    """Build supervised sequences per machine to avoid leakage across equipment."""
    X_all, y_all = [], []

    for _, group in df.groupby("machine_id"):
        features = group[SENSOR_COLS].values
        targets = group["failure"].values
        if len(features) < window_size + horizon:
            continue
        X, y = create_sequences(features, targets, window_size, horizon)
        X_all.append(X)
        y_all.append(y)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    val_ratio = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    # Scale per-feature across all timesteps
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def scale(X_arr: np.ndarray) -> np.ndarray:
        shape = X_arr.shape
        return scaler.transform(X_arr.reshape(-1, n_features)).reshape(shape)

    return {
        "X_train": scale(X_train),
        "X_val": scale(X_val),
        "X_test": scale(X_test),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "n_features": n_features,
        "window_size": window_size,
    }


def prepare_autoencoder_data(
    df: pd.DataFrame,
    test_split: float = 0.2,
) -> dict:
    """Use normal (non-failure) windows for autoencoder training."""
    normal = df[df["failure"] == 0]
    features = normal[SENSOR_COLS].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    X_train, X_test = train_test_split(scaled, test_size=test_split, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "scaler": scaler}


def save_processed_artifacts(data: dict, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        if key in data:
            np.save(output_path / f"{key}.npy", data[key])
    if "scaler" in data:
        joblib.dump(data["scaler"], output_path / "scaler.joblib")


def run_preprocessing(config_path: str = "config/config.yaml") -> dict:
    config = load_config(config_path)
    df = load_raw_data(config["data"]["raw_dir"])
    feat_cfg = config["features"]

    failure_data = prepare_failure_prediction_data(
        df,
        window_size=feat_cfg["window_size"],
        horizon=feat_cfg["forecast_horizon"],
        test_split=feat_cfg["test_split"],
        val_split=feat_cfg["val_split"],
    )
    ae_data = prepare_autoencoder_data(df)

    processed_dir = Path(config["data"]["processed_dir"])
    save_processed_artifacts(failure_data, processed_dir)
    joblib.dump(ae_data["scaler"], processed_dir / "ae_scaler.joblib")
    np.save(processed_dir / "ae_X_train.npy", ae_data["X_train"])
    np.save(processed_dir / "ae_X_test.npy", ae_data["X_test"])

    return {"failure": failure_data, "autoencoder": ae_data}
