"""Autoencoder-based anomaly scoring and detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import joblib
import tensorflow as tf

from src.models.autoencoder import fit_threshold, reconstruction_errors


class AutoencoderAnomalyDetector:
    def __init__(self, model: tf.keras.Model, threshold: float, scaler=None):
        self.model = model
        self.threshold = threshold
        self.scaler = scaler

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return reconstruction_errors(self.model, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.threshold).astype(int)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(model_dir / "autoencoder_detector.keras")
        meta = {"threshold": self.threshold}
        with open(model_dir / "autoencoder_meta.json", "w") as f:
            json.dump(meta, f)
        if self.scaler is not None:
            joblib.dump(self.scaler, model_dir / "ae_scaler.joblib")

    @classmethod
    def load(cls, model_dir: str | Path) -> "AutoencoderAnomalyDetector":
        model_dir = Path(model_dir)
        model = tf.keras.models.load_model(model_dir / "autoencoder_detector.keras")
        with open(model_dir / "autoencoder_meta.json") as f:
            meta = json.load(f)
        scaler_path = model_dir / "ae_scaler.joblib"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        return cls(model=model, threshold=meta["threshold"], scaler=scaler)


def build_detector_from_training(
    model: tf.keras.Model,
    X_train: np.ndarray,
    percentile: float = 95.0,
    scaler=None,
) -> AutoencoderAnomalyDetector:
    errors = reconstruction_errors(model, X_train)
    threshold = fit_threshold(errors, percentile)
    return AutoencoderAnomalyDetector(model=model, threshold=threshold, scaler=scaler)
