from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import Model


def load_registered_model(model_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    version = client.get_model_version_by_alias(model_name, "champion")
    model_uri = f"models:/{model_name}/{version.version}"
    model_directory = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    metadata = dict(Model.load(str(Path(model_directory) / "MLmodel")).metadata or {})
    if not metadata.get("features"):
        raise ValueError(f"Model {model_uri} has no feature list in metadata")
    pipeline = mlflow.sklearn.load_model(model_directory)
    metadata["model_version"] = str(version.version)
    return {"pipeline": pipeline, "metadata": metadata, "model_uri": model_uri}
