"""Log and register the frozen ensemble. DagsHub if URI is set, else local sqlite mlflow.db."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from smart_grid.config import load_params, mlflow_tracking_uri, models_dir, project_root
from smart_grid.features.build import FEATURE_COLS

logger = logging.getLogger(__name__)


def _configure_mlflow() -> str:
    uri = mlflow_tracking_uri()
    user = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    password = (
        os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
        or os.getenv("DAGSHUB_TOKEN", "").strip()
    )
    if user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    if not uri:
        db_path = (project_root() / "mlflow.db").resolve().as_posix()
        uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    params = load_params()
    experiment = params.get("mlflow", {}).get("experiment", "peco-demand")
    mlflow.set_experiment(experiment)
    return uri


def register_ensemble(bundle: dict, sample_x: pd.DataFrame | None = None) -> dict[str, Any]:
    params = load_params()
    uri = _configure_mlflow()
    registered_name = params.get("mlflow", {}).get("registered_model", "peco-demand-ensemble")
    alias = params.get("model", {}).get("alias", "production")
    meta = bundle["metadata"]
    sample = sample_x if sample_x is not None else pd.DataFrame(columns=FEATURE_COLS)

    with mlflow.start_run(run_name="freeze-ensemble-mean") as run:
        mlflow.log_param("winner", meta["winner"])
        mlflow.log_param("horizon_hours", meta["horizon_hours"])
        mlflow.log_param("n_train", meta["n_train"])
        for key, value in meta["lightgbm_params"].items():
            mlflow.log_param(f"lgbm_{key}", value)
        for key, value in meta["xgboost_params"].items():
            mlflow.log_param(f"xgb_{key}", value)

        for split_name, scores in meta["refit_metrics"].items():
            for metric_name, metric_value in scores.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", float(metric_value))

        colab = meta["colab_selection_metrics"]["ensemble_mean"]
        for metric_name, metric_value in colab.items():
            mlflow.log_metric(f"colab_test_{metric_name}", float(metric_value))

        pin = models_dir()
        if (pin / "metadata.json").exists():
            mlflow.log_artifacts(str(pin), artifact_path="production_pin")

        signature = None
        if len(sample):
            preds = bundle["ensemble"].predict(sample)
            signature = infer_signature(sample, preds)

        mlflow.sklearn.log_model(
            bundle["ensemble"],
            name="model",
            signature=signature,
            registered_model_name=registered_name,
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id

    version = _alias_latest(registered_name, alias)
    logger.info(
        "MLflow run=%s uri=%s model=%s version=%s alias=%s",
        run_id,
        uri,
        registered_name,
        version,
        alias,
    )
    return {
        "tracking_uri": uri,
        "run_id": run_id,
        "registered_model": registered_name,
        "version": version,
        "alias": alias,
    }


def _alias_latest(name: str, alias: str) -> str | None:
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    try:
        client.set_registered_model_alias(name, alias, latest.version)
    except Exception:
        # Older MLflow / DagsHub: fall back to the classic Production stage.
        try:
            client.transition_model_version_stage(
                name=name,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
        except Exception as exc:
            logger.warning("Could not set alias/stage on %s: %s", name, exc)
    return str(latest.version)
