import json
import math
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from mlops_playground.api import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(create_app(ROOT / "artifact/model.joblib")) as client:
        yield client


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples/predict.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("body", [b"not-json", b"{", b"[]", b"null"])
def test_contract_garbage_returns_422(client, body):
    response = client.post(
        "/v1/predict", content=body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"]


def test_contract_extra_field_returns_422(client, payload):
    response = client.post("/v1/predict", json={**payload, "unexpected_field": 123})
    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden"
        and error["loc"] == ["body", "unexpected_field"]
        for error in response.json()["detail"]
    )


def test_contract_wrong_type_returns_422(client, payload):
    response = client.post("/v1/predict", json={**payload, "month": "garbage"})
    assert response.status_code == 422
    assert any(
        error["type"] == "int_type" and error["loc"] == ["body", "month"]
        for error in response.json()["detail"]
    )


def test_smoke_response_range_and_types(client, payload):
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version", "request_id", "latency_ms"}
    assert type(body["prediction"]) is float
    assert math.isfinite(body["prediction"]) and body["prediction"] >= 0
    assert type(body["model_version"]) is str and body["model_version"]
    assert type(body["request_id"]) is str
    assert UUID(body["request_id"]).version == 4
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert type(body["latency_ms"]) is float
    assert math.isfinite(body["latency_ms"]) and body["latency_ms"] >= 0


def test_smoke_missing_features(client):
    payload = json.loads((ROOT / "examples/predict_missing.json").read_text(encoding="utf-8"))
    assert any(value is None for value in payload.values())
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert type(prediction) is float
    assert math.isfinite(prediction) and prediction >= 0


def test_determinism_same_input_same_prediction(client, payload):
    first = client.post("/v1/predict", json=payload)
    second = client.post("/v1/predict", json=payload)
    assert first.status_code == second.status_code == 200
    first_body, second_body = first.json(), second.json()
    assert first_body["prediction"] == second_body["prediction"]
    assert first_body["model_version"] == second_body["model_version"]
    assert first_body["request_id"] != second_body["request_id"]
