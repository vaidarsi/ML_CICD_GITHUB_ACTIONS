"""Tests for the FastAPI endpoints."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app
from src import config


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs the startup (model loading)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features() -> list[float]:
    df = pd.read_csv(config.RAW_DATA_PATH)

    return df.drop(columns=[config.TARGET_COLUMN]).iloc[0].tolist()


def test_root(client):
    r = client.get("/")

    assert r.status_code == 200
    assert "model_version" in r.json()


def test_health_when_model_loaded(client):
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_predict_valid_input(client, sample_features):
    r = client.post(
        "/predict",
        json={"features": sample_features},
    )

    assert r.status_code == 200

    body = r.json()

    assert body["prediction"] in {
        "class_0",
        "class_1",
        "class_2",
    }

    assert "model_version" in body


def test_predict_too_few_features(client):
    r = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert r.status_code == 422


def test_predict_non_numeric_features(client, sample_features):
    bad = ["abc"] + sample_features[1:]

    r = client.post(
        "/predict",
        json={"features": bad},
    )

    assert r.status_code == 422


def test_predict_missing_body_field(client):
    r = client.post(
        "/predict",
        json={},
    )

    assert r.status_code == 422


def test_metrics_endpoint(client, sample_features):
    client.post(
        "/predict",
        json={"features": sample_features},
    )

    r = client.get("/metrics")

    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "predictions_total" in r.text
    assert "http_errors_total" in r.text


def test_health_unhealthy_when_model_missing(client):
    saved = main.state["model"]
    main.state["model"] = None

    try:
        assert client.get("/health").status_code == 503

        r = client.post(
            "/predict",
            json={"features": [0.0] * 13},
        )

        assert r.status_code == 503

    finally:
        main.state["model"] = saved
