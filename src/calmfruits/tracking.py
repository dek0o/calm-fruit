"""Small explicit MLflow integration for reproducible CalmFruits runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


EXPERIMENT_NAME = "calmfruits_semantic_search"


def configure_mlflow() -> MlflowClient:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if not uri or not username or not password:
        raise RuntimeError("MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD are required")
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL")
    if s3_endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def experiment_id(client: MlflowClient) -> str:
    existing = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(EXPERIMENT_NAME)


def log_evaluation_run(
    run_name: str,
    params: dict[str, Any],
    summary: pd.DataFrame,
    artifacts: list[Path],
) -> str:
    client = configure_mlflow()
    exp_id = experiment_id(client)
    row = summary.iloc[0].to_dict()
    metric_values = {key: float(value) for key, value in row.items() if key != "method" and pd.notna(value)}
    safe_params = {key: str(value) for key, value in params.items()}
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.log_params(safe_params)
        mlflow.log_metrics(metric_values)
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="artifacts")
        return run.info.run_id
