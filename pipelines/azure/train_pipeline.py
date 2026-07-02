#!/usr/bin/env python3
"""
Azure ML pipeline definition for automated training and batch inference.

Requires: az login && populated .env with AZURE_SUBSCRIPTION_ID
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

try:
    from azure.ai.ml import MLClient, Input, Output, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.dsl import pipeline
    from azure.identity import DefaultAzureCredential
except ImportError as e:
    raise SystemExit(
        "Azure SDK not installed. Run: pip install azure-ai-ml azure-identity"
    ) from e


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_ml_client(config: dict) -> MLClient:
    azure_cfg = config["azure"]
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID") or azure_cfg["subscription_id"]
    if subscription_id.startswith("${"):
        raise ValueError("Set AZURE_SUBSCRIPTION_ID in .env before submitting pipeline")

    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=azure_cfg["resource_group"],
        workspace_name=azure_cfg["workspace_name"],
    )


@pipeline(
    name="predictive_maintenance_pipeline",
    description="IoT sensor data ingestion, model training, versioning, and batch inference",
)
def predictive_maintenance_pipeline(
    raw_data: Input,
    model_type: str = "lstm",
):
    """Azure ML pipeline: train models and run scheduled batch inference."""
    train_step = command(
        name="train_models",
        display_name="Train LSTM/GRU and Autoencoder",
        code=str(Path(__file__).parent.parent),
        command="python pipelines/local_train.py --model all --skip-data-gen",
        environment="pred-maint-env@latest",
        compute="cpu-cluster",
        inputs={"raw_data": raw_data},
        outputs={"model_output": Output(type=AssetTypes.MLFLOW_MODEL)},
    )

    infer_step = command(
        name="batch_inference",
        display_name="Scheduled Batch Inference",
        code=str(Path(__file__).parent.parent),
        command=f"python pipelines/local_inference.py --model {model_type}",
        environment="pred-maint-env@latest",
        compute="cpu-cluster",
        inputs={"model_input": train_step.outputs.model_output},
        outputs={"predictions": Output(type=AssetTypes.URI_FOLDER)},
    )

    return {"model": train_step.outputs.model_output, "predictions": infer_step.outputs.predictions}


def submit_pipeline(config_path: str = "config/config.yaml") -> None:
    config = load_config(config_path)
    ml_client = get_ml_client(config)

    raw_data_path = Path(config["data"]["raw_dir"]) / "iot_sensor_data.csv"
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Upload raw data first or run local training to generate {raw_data_path}"
        )

    pipeline_job = predictive_maintenance_pipeline(
        raw_data=Input(type=AssetTypes.URI_FILE, path=str(raw_data_path))
    )
    pipeline_job.settings.default_compute = config["azure"]["compute_cluster"]

    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name=config["azure"]["experiment_name"],
    )
    print(f"Pipeline submitted: {returned_job.studio_url}")


if __name__ == "__main__":
    submit_pipeline()
