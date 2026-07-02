#!/usr/bin/env python3
"""End-to-end local training: data generation, preprocessing, LSTM/GRU, autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.generate_synthetic_data import generate_dataset, save_dataset
from src.data.preprocessing import run_preprocessing, load_config
from src.models.sequence_models import build_lstm_model, build_gru_model, train_sequence_model
from src.models.autoencoder import build_autoencoder, train_autoencoder
from src.anomaly.autoencoder_detector import build_detector_from_training
from src.anomaly.rolling_window import detect_multivariate_anomalies
from src.data.preprocessing import load_raw_data, SENSOR_COLS
from src.utils.metrics import classification_metrics, save_metrics


def train_failure_model(model_type: str, data: dict, config: dict) -> dict:
    cfg = config["models"][model_type]
    input_shape = (data["window_size"], data["n_features"])

    if model_type == "lstm":
        model = build_lstm_model(input_shape, cfg["units"], cfg["dropout"], cfg["learning_rate"])
    else:
        model = build_gru_model(input_shape, cfg["units"], cfg["dropout"], cfg["learning_rate"])

    model = train_sequence_model(
        model,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        model_dir=config["paths"]["models_dir"],
        model_name=model_type,
    )

    y_prob = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = classification_metrics(data["y_test"], y_pred, y_prob)
    save_metrics(metrics, Path(config["paths"]["artifacts_dir"]) / f"{model_type}_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", choices=["lstm", "gru", "both", "autoencoder", "all"], default="all")
    parser.add_argument("--skip-data-gen", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = Path(config["data"]["raw_dir"]) / "iot_sensor_data.csv"

    if not args.skip_data_gen or not raw_path.exists():
        print("Generating synthetic IoT sensor data...")
        df = generate_dataset(config)
        save_dataset(df, config["data"]["raw_dir"])

    print("Preprocessing...")
    processed = run_preprocessing(args.config)
    failure_data = processed["failure"]

    results = {}

    if args.model in ("lstm", "both", "all"):
        print("\nTraining LSTM...")
        results["lstm"] = train_failure_model("lstm", failure_data, config)

    if args.model in ("gru", "both", "all"):
        print("\nTraining GRU...")
        results["gru"] = train_failure_model("gru", failure_data, config)

    if args.model in ("autoencoder", "all"):
        print("\nTraining autoencoder...")
        ae_cfg = config["models"]["autoencoder"]
        ae_data = processed["autoencoder"]
        split = int(len(ae_data["X_train"]) * 0.9)
        ae_model = build_autoencoder(
            input_dim=ae_data["X_train"].shape[1],
            encoding_dim=ae_cfg["encoding_dim"],
            learning_rate=ae_cfg["learning_rate"],
        )
        ae_model = train_autoencoder(
            ae_model,
            ae_data["X_train"][:split],
            ae_data["X_train"][split:],
            epochs=ae_cfg["epochs"],
            batch_size=ae_cfg["batch_size"],
            model_dir=config["paths"]["models_dir"],
        )
        detector = build_detector_from_training(
            ae_model,
            ae_data["X_train"],
            percentile=ae_cfg["anomaly_threshold_percentile"],
            scaler=ae_data["scaler"],
        )
        detector.save(config["paths"]["models_dir"])
        ae_scores = detector.score(ae_data["X_test"])
        results["autoencoder"] = {
            "threshold": detector.threshold,
            "mean_reconstruction_error": float(np.mean(ae_scores)),
        }

    print("\nRunning rolling-window anomaly detection...")
    df = load_raw_data(config["data"]["raw_dir"])
    anomaly_df = detect_multivariate_anomalies(
        df,
        SENSOR_COLS,
        window=config["anomaly"]["rolling_window"],
        threshold=config["anomaly"]["z_score_threshold"],
    )
    anomaly_rate = float(anomaly_df["rolling_anomaly"].mean())
    results["rolling_window"] = {"anomaly_rate": anomaly_rate}
    anomaly_df.to_csv(Path(config["paths"]["artifacts_dir"]) / "rolling_anomalies.csv", index=False)

    summary_path = Path(config["paths"]["artifacts_dir"]) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete. Summary saved to {summary_path}")
    for name, metrics in results.items():
        if "f1" in metrics:
            print(f"  {name}: F1={metrics['f1']:.3f}, AUC={metrics.get('roc_auc', 0):.3f}")


if __name__ == "__main__":
    main()
