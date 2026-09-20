import json
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest

from mlops_playground import api
from mlops_playground.pipeline import daily_input
from mlops_playground.schemas import PredictRequest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifact/model.joblib"


@pytest.fixture(scope="module")
def payload():
    return json.loads((ROOT / "examples/predict.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(api.create_app(ARTIFACT)) as client:
        yield client


def test_status_and_docs(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "model_version": "1.0.0"}
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    request = schema["components"]["schemas"]["PredictRequest"]
    assert request["additionalProperties"] is False
    assert request["properties"]["month"]["maximum"] == 12
    assert "month" in request["required"]
    assert "lag_7" not in request["required"]
    assert client.post("/v1/predict", json=request["examples"][0]).status_code == 200


def test_prediction_matches_original_pipeline(client, payload):
    bundle = joblib.load(ARTIFACT)
    metadata = bundle["metadata"]
    frame = daily_input(metadata["history"], metadata["raw_daily_features"])
    expected = bundle["pipeline"].predict(frame)[-1]
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"prediction", "model_version", "request_id", "latency_ms"}
    assert body["prediction"] == pytest.approx(expected)
    assert body["model_version"] == metadata["version"]
    assert UUID(body["request_id"]).version == 4
    assert body["latency_ms"] >= 0
    again = client.post("/v1/predict", json=dict(reversed(list(payload.items())))).json()
    assert again["prediction"] == body["prediction"]
    assert again["request_id"] != body["request_id"]


def test_real_missing_features_and_omitted_fields(client):
    payload = json.loads((ROOT / "examples/predict_missing.json").read_text(encoding="utf-8"))
    explicit = client.post("/v1/predict", json=payload)
    omitted = client.post("/v1/predict", json={k: v for k, v in payload.items() if v is not None})
    assert explicit.status_code == omitted.status_code == 200
    assert explicit.json()["prediction"] == omitted.json()["prediction"]


@pytest.mark.parametrize("field,value", [
    ("extra_field", 1), ("month", 0), ("month", 13),
    ("day_of_week", -1), ("day_of_week", 7), ("is_weekend", 2),
    ("dow_sin", -1.01), ("year_cos", 1.01), ("month", None),
    ("month", "12"), ("is_weekend", True), ("lag_1", -1),
    ("lag_1", 1.5), ("raw_rows_count", -1), ("raw_price_mean", -1),
    ("rolling_std_7", -1), ("raw_Month_mean", 13), ("raw_Year_min", 0),
])
def test_invalid_features(client, payload, field, value):
    response = client.post("/v1/predict", json={**payload, field: value})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_required_field(client, payload):
    response = client.post("/v1/predict", json={k: v for k, v in payload.items() if k != "month"})
    assert response.status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_returns_json_error(client, payload, value):
    response = client.post(
        "/v1/predict", content=json.dumps({**payload, "raw_grand_total_mean": value}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_negative_amounts_seen_in_data_are_valid(client, payload):
    response = client.post("/v1/predict", json={
        **payload, "raw_grand_total_min": -1594, "raw_discount_amount_min": -599.5,
    })
    assert response.status_code == 200


def test_model_loaded_once_per_lifespan(payload):
    with patch.object(api.joblib, "load", wraps=joblib.load) as loader:
        app = api.create_app(ARTIFACT)
        loader.assert_not_called()
        with TestClient(app) as client:
            for _ in range(3):
                assert client.post("/v1/predict", json=payload).status_code == 200
                assert client.get("/ready").status_code == 200
            loader.assert_called_once_with(ARTIFACT)
        assert app.state.model is None


@pytest.mark.parametrize("failure", ["missing", "corrupt", "schema", "version"])
def test_unavailable_model(tmp_path, payload, failure):
    path = tmp_path / "model.joblib"
    if failure == "corrupt":
        path.write_bytes(b"not a model")
    elif failure in {"schema", "version"}:
        bundle = joblib.load(ARTIFACT)
        if failure == "schema":
            bundle["metadata"]["feature_columns"] = ["wrong"]
        else:
            bundle["metadata"]["sklearn_version"] = "0.0.0"
        joblib.dump(bundle, path)
    with TestClient(api.create_app(path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/predict", json=payload).status_code == 503


def test_all_historical_feature_rows_match_schema():
    bundle = joblib.load(ARTIFACT)
    metadata = bundle["metadata"]
    frame = daily_input(metadata["history"], metadata["raw_daily_features"])
    features = bundle["pipeline"].named_steps["preprocessing"].transform(frame)
    properties = PredictRequest.model_json_schema()["properties"]
    assert list(features.columns) == list(properties)
    for _, row in features.iterrows():
        payload = {}
        for name, value in row.items():
            prop = properties[name]
            kind = prop.get("type") or prop["anyOf"][0]["type"]
            if pd.isna(value):
                payload[name] = None
            elif kind == "integer":
                assert float(value).is_integer()
                payload[name] = int(value)
            else:
                payload[name] = float(value)
        PredictRequest.model_validate(payload)


def test_bad_prediction_does_not_leak_exception(payload):
    with TestClient(api.create_app(ARTIFACT)) as client:
        with patch.object(client.app.state.model.estimator, "predict", return_value=np.array([np.nan])):
            response = client.post("/v1/predict", json=payload)
        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction failed"}


def test_model_path_environment(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(ARTIFACT))
    with TestClient(api.create_app()) as client:
        assert client.get("/ready").status_code == 200
