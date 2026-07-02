"""Dense autoencoder for unsupervised anomaly detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def build_autoencoder(
    input_dim: int,
    encoding_dim: int = 16,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation="relu")(inputs)
    encoded = layers.Dense(encoding_dim, activation="relu")(encoded)
    decoded = layers.Dense(64, activation="relu")(encoded)
    outputs = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(inputs, outputs, name="sensor_autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder


def reconstruction_errors(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    reconstructions = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructions), axis=1)


def fit_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(errors, percentile))


def train_autoencoder(
    model: tf.keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    epochs: int = 25,
    batch_size: int = 128,
    model_dir: str | Path = "models",
) -> tf.keras.Model:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    val_data = (X_val, X_val) if X_val is not None else None
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        X_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    model.save(model_dir / "autoencoder_final.keras")
    return model
