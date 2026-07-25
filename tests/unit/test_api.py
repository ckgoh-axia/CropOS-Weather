import torch
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app

client = TestClient(app)

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_forecast_per_horizon():
    with patch("src.api.main.run_inference", return_value=torch.tensor([[0.1, 0.65, 0.8, 0.3]])):
        response = client.post("/predict", json={
            "farm_id": "farm_001",
            "lat": 15.25, "lon": 104.87,
            "timestamp": "2023-06-15T00:00:00Z",
        })
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 4  # 4 horizons
    assert data["forecast"][0]["horizon_hours"] == 12
    assert 0.0 <= data["forecast"][0]["rain_probability"] <= 1.0

def test_predict_out_of_bounds_lat_returns_422():
    response = client.post("/predict", json={
        "farm_id": "farm_001", "lat": 99.0, "lon": 104.87,
        "timestamp": "2023-06-15T00:00:00Z",
    })
    assert response.status_code == 422

def test_predict_alert_flag_set_above_threshold():
    # 0.9 probability should trigger alert
    with patch("src.api.main.run_inference", return_value=torch.tensor([[0.9, 0.9, 0.9, 0.9]])):
        response = client.post("/predict", json={
            "farm_id": "farm_001", "lat": 15.0, "lon": 102.0,
            "timestamp": "2023-06-15T00:00:00Z",
        })
    assert all(h["alert"] for h in response.json()["forecast"])
