"""Synthetic multivariate IoT sensor data for predictive maintenance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_machine_series(
    machine_id: int,
    n_samples: int,
    rng: np.random.Generator,
    failure_rate: float,
) -> pd.DataFrame:
    """Simulate degrading equipment with temperature, vibration, and pressure."""
    t = np.arange(n_samples)
    health = np.linspace(1.0, 0.0, n_samples)

    # Baseline operating conditions with drift as health degrades
    temperature = 65 + 15 * (1 - health) + rng.normal(0, 1.5, n_samples)
    vibration = 2.5 + 8 * (1 - health) ** 1.5 + rng.normal(0, 0.3, n_samples)
    pressure = 100 - 20 * (1 - health) + rng.normal(0, 2.0, n_samples)

    # Inject failure events near end of life
    failure_window = max(1, int(n_samples * failure_rate))
    failure_label = np.zeros(n_samples, dtype=int)
    failure_label[-failure_window:] = 1

    # Spike anomalies before failure
    anomaly = np.zeros(n_samples, dtype=int)
    pre_failure = max(0, n_samples - failure_window - 50)
    anomaly_indices = rng.choice(
        range(pre_failure, n_samples - failure_window),
        size=min(5, max(1, (n_samples - failure_window - pre_failure) // 10)),
        replace=False,
    )
    for idx in anomaly_indices:
        vibration[idx] += rng.uniform(3, 6)
        temperature[idx] += rng.uniform(5, 12)
        anomaly[idx] = 1

    return pd.DataFrame(
        {
            "machine_id": machine_id,
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="5min"),
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "health_index": health,
            "failure": failure_label,
            "anomaly": anomaly,
        }
    )


def generate_dataset(config: dict | None = None) -> pd.DataFrame:
    cfg = config or _load_config()
    data_cfg = cfg["data"]
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    frames = [
        generate_machine_series(
            machine_id=i,
            n_samples=data_cfg["samples_per_machine"],
            rng=rng,
            failure_rate=data_cfg["failure_rate"],
        )
        for i in range(data_cfg["n_machines"])
    ]
    return pd.concat(frames, ignore_index=True)


def save_dataset(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "iot_sensor_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IoT sensor data")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = _load_config(args.config)
    df = generate_dataset(config)
    path = save_dataset(df, config["data"]["raw_dir"])
    print(f"Generated {len(df):,} rows across {config['data']['n_machines']} machines")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
