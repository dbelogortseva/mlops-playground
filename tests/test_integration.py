import os

import psycopg
import pytest

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="PostgreSQL is required: set DATABASE_URL",
    ),
]


def test_prediction_is_logged(client, good_row):
    response = client.post("/v1/predict", json=good_row)
    body = response.json()
    assert response.headers["X-Request-ID"] == body["request_id"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, prediction, features, latency_ms, "
            "status_code FROM predictions WHERE request_id = %s",
            (body["request_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["prediction"])
    assert row[2] == good_row
    assert row[3] == pytest.approx(body["latency_ms"], abs=0.01)
    assert row[4] == response.status_code


def test_validation_error_is_logged(client, good_row):
    invalid_row = {**good_row, "hacker_field": 1}
    response = client.post(
        "/v1/predict",
        json=invalid_row,
    )
    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT features, prediction, status_code FROM predictions "
            "WHERE request_id = %s",
            (request_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == invalid_row
    assert row[1] is None
    assert row[2] == 422
