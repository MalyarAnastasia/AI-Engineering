import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    from src.service.app import app, load_models
    load_models()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["kmeans_loaded"] is True


def test_predict_endpoint(client):
    payload = {
        "area": 65.5,
        "kitchen_area": 10.0,
        "rooms": 2,
        "geo_lat": 55.7558,
        "geo_lon": 37.6173,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] > 0
    assert data["price_per_meter"] > 0
    assert data["investment_rating"] in {"undervalued", "fair", "overvalued"}
    assert data["cluster"] is not None
