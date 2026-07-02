#!/usr/bin/env python3
"""Batch inference for failure prediction and anomaly scoring."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yaml

from src.data.preprocessing import SENSOR_COLS, create_sequences
from src.anomaly.autoencoder_detector import AutoencoderAnomalyDetector
from src.anomaly.rolling_window import detect_multivariate_anomalies


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_batch_inference(
    input_csv: str | Path,
    config: dict,
    model_type: str = "lstm",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    models_dir = Path(config["paths"]["models_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Rolling-window anomalies
    df = detect_multivariate_anomalies(
        df,
        SENSOR_COLS,
        window=config["anomaly"]["rolling_window"],
        threshold=config["anomaly"]["z_score_threshold"],
    )

    # Autoencoder scores on raw sensor readings
    ae_detector = AutoencoderAnomalyDetector.load(models_dir)
    sensor_matrix = df[SENSOR_COLS].values
    df["ae_reconstruction_error"] = ae_detector.score(sensor_matrix)
    df["ae_anomaly"] = (df["ae_reconstruction_error"] > ae_detector.threshold).astype(int)

    # Sequence model failure probability (per machine)
    model_path = models_dir / f"{model_type}_best.keras"
    if not model_path.exists():
        model_path = models_dir / f"{model_type}_final.keras"

    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(Path(config["data"]["processed_dir"]) / "scaler.joblib")
        window = config["features"]["window_size"]
        horizon = config["features"]["forecast_horizon"]

        failure_probs = []
        for _, group in df.groupby("machine_id"):
            features = group[SENSOR_COLS].values
            targets = np.zeros(len(features))  # placeholder for sequence creation
            if len(features) < window + horizon:
                failure_probs.extend([np.nan] * len(group))
                continue
            X, _ = create_sequences(
                scaler.transform(features), targets, window, horizon
            )
            probs = model.predict(X, verbose=0).flatten()
            padded = [np.nan] * (window + horizon - 1) + list(probs)
            failure_probs.extend(padded[: len(group)])

        df[f"{model_type}_failure_probability"] = failure_probs
        df[f"{model_type}_failure_alert"] = (df[f"{model_type}_failure_probability"] >= 0.5).astype(float)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = artifacts_dir / f"inference_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    summary = {
        "timestamp": timestamp,
        "rows": len(df),
        "rolling_anomalies": int(df["rolling_anomaly"].sum()),
        "ae_anomalies": int(df["ae_anomaly"].sum()),
        "output_file": str(output_path),
    }
    with open(artifacts_dir / f"inference_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default=None, help="Input CSV (defaults to raw data)")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    args = parser.parse_args()

    config = load_config(args.config)
    input_csv = args.input or Path(config["data"]["raw_dir"]) / "iot_sensor_data.csv"

    print(f"Running batch inference on {input_csv}...")
    result = run_batch_inference(input_csv, config, args.model)
    print(f"Inference complete. {len(result):,} rows scored.")


if __name__ == "__main__":
    main()
