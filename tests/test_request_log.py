import json
import os
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient
import numpy as np
import psycopg
import pytest

from mlops_playground.api import create_app
from mlops_playground.request_log import PostgresRequestLog, request_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifact/model.joblib"
PAYLOAD = json.loads((ROOT / "examples/predict.json").read_text(encoding="utf-8"))


def test_without_database_no_connection():
    with patch("mlops_playground.request_log.psycopg.connect") as connect:
        with TestClient(create_app(ARTIFACT)) as client:
            response = client.post("/v1/predict", json=PAYLOAD)
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == response.json()["request_id"]
        connect.assert_not_called()


def test_request_records_include_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    with patch.object(PostgresRequestLog, "initialize") as initialize, patch.object(PostgresRequestLog, "write") as write:
        with TestClient(create_app(ARTIFACT)) as client:
            responses = [
                client.post("/v1/predict", json=PAYLOAD),
                client.post("/v1/predict", json={**PAYLOAD, "month": 13}),
                client.post("/v1/predict", json={**PAYLOAD, "unexpected": 1}),
                client.post("/v1/predict", content=b"{broken"),
                client.get("/v1/predict"),
            ]
            with patch.object(client.app.state.model.estimator, "predict", return_value=np.array([np.nan])):
                responses.append(client.post("/v1/predict", json=PAYLOAD))
            client.get("/health")
            client.get("/docs")
        with TestClient(create_app(tmp_path / "missing.joblib")) as client:
            responses.append(client.post("/v1/predict", json=PAYLOAD))
        records = [call.args[0] for call in write.call_args_list]
        assert initialize.call_count == 2
        assert [r.status_code for r in records] == [200, 422, 422, 422, 405, 500, 503]
        assert len({r.request_id for r in records}) == 7
        for record, response in zip(records, responses, strict=True):
            assert str(record.request_id) == response.headers["X-Request-ID"]
            assert record.requested_at.utcoffset().total_seconds() == 0
            assert record.latency_ms >= 0
        successful = responses[0].json()
        assert records[0].features == PAYLOAD
        assert records[0].prediction == successful["prediction"]
        assert records[0].latency_ms == successful["latency_ms"]
        assert records[0].model_version == successful["model_version"]
        assert all(r.prediction is None for r in records[1:])
        assert records[-1].model_version is None
        assert records[1].features["month"] == 13
        assert bytes.fromhex(records[3].features["_raw_body_hex"]) == b"{broken"


def test_database_failure_preserves_response(monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    with patch.object(PostgresRequestLog, "initialize", side_effect=RuntimeError), patch.object(PostgresRequestLog, "write", side_effect=RuntimeError):
        with TestClient(create_app(ARTIFACT)) as client:
            assert client.post("/v1/predict", json=PAYLOAD).status_code == 200
    assert "Request log initialization failed" in caplog.text
    assert "Request log write failed" in caplog.text


@pytest.mark.parametrize("body", [b"{", b"", b'NaN', b'{"x": 1e999}', b'{"x": "\\u0000"}', b'{"x": "\\ud800"}', b"\xff"])
def test_invalid_json_is_preserved_as_hex(body):
    assert request_features(body) == {"_raw_body_hex": body.hex(), "_encoding": "hex"}


def test_valid_json_is_preserved():
    assert request_features(b'{"month": 12, "lag_1": null}') == {"month": 12, "lag_1": None}


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="Set TEST_DATABASE_URL for a real PostgreSQL check")
def test_real_postgres(monkeypatch, tmp_path):
    url = os.environ["TEST_DATABASE_URL"]
    monkeypatch.setenv("DATABASE_URL", url)
    with TestClient(create_app(ARTIFACT)) as client:
        responses = [
            client.post("/v1/predict", json=PAYLOAD),
            client.post("/v1/predict", json={**PAYLOAD, "month": 13}),
            client.post("/v1/predict", json={**PAYLOAD, "unexpected": 1}),
            client.post("/v1/predict", content=b"{broken"),
        ]
        with patch.object(client.app.state.model.estimator, "predict", return_value=np.array([np.nan])):
            responses.append(client.post("/v1/predict", json=PAYLOAD))
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        responses.append(client.post("/v1/predict", json=PAYLOAD))
    assert [response.status_code for response in responses] == [200, 422, 422, 422, 500, 503]
    with psycopg.connect(url) as connection:
        for response in responses:
            row = connection.execute(
                "SELECT request_id, requested_at, model_version, features, prediction, latency_ms, status_code, pg_typeof(features)::text FROM prediction_requests WHERE request_id = %s",
                (UUID(response.headers["X-Request-ID"]),),
            ).fetchone()
            assert row is not None
            assert row[1].tzinfo is not None
            assert row[5] >= 0
            assert row[6] == response.status_code
            assert row[7] == "jsonb"
            if response.status_code == 200:
                body = response.json()
                assert row[2] == body["model_version"]
                assert row[3] == PAYLOAD
                assert row[4] == body["prediction"]
                assert row[5] == body["latency_ms"]
            else:
                assert row[4] is None
            if response.status_code == 503:
                assert row[2] is None
