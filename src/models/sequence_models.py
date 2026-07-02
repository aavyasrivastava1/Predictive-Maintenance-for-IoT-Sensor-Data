"""LSTM and GRU sequence classifiers for failure prediction."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def build_lstm_model(
    input_shape: tuple[int, int],
    units: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    units = units or [64, 32]
    model = models.Sequential(name="lstm_failure_predictor")
    model.add(layers.Input(shape=input_shape))
    for i, u in enumerate(units):
        return_sequences = i < len(units) - 1
        model.add(layers.LSTM(u, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_gru_model(
    input_shape: tuple[int, int],
    units: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    units = units or [64, 32]
    model = models.Sequential(name="gru_failure_predictor")
    model.add(layers.Input(shape=input_shape))
    for i, u in enumerate(units):
        return_sequences = i < len(units) - 1
        model.add(layers.GRU(u, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_sequence_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 30,
    batch_size: int = 64,
    model_dir: str | Path = "models",
    model_name: str = "model",
) -> tf.keras.Model:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(model_dir / f"{model_name}_best.keras"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )
    model.save(model_dir / f"{model_name}_final.keras")
    return model
