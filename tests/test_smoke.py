import math


def test_predict_smoke(client, good_row):
    response = client.post("/v1/predict", json=good_row)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])
    assert body["prediction"] >= 0
    assert isinstance(body["model_version"], str)
    assert isinstance(body["request_id"], str)
    assert isinstance(body["latency_ms"], float)
    assert body["latency_ms"] >= 0


def test_predict_handles_missing_value(client, good_row):
    response = client.post(
        "/v1/predict",
        json={**good_row, "raw_price_mean": None},
    )
    assert response.status_code == 200


def test_same_input_gives_same_prediction(client, good_row):
    first = client.post("/v1/predict", json=good_row).json()["prediction"]
    second = client.post("/v1/predict", json=good_row).json()["prediction"]
    assert abs(first - second) < 1e-12
