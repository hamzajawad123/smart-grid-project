from fastapi.testclient import TestClient

from smart_grid.api.app import create_app


def test_health_returns_json():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["grid_id"] == "PJM_PE"
    assert "model_loaded" in body


def test_forecast_rejects_other_grid():
    client = TestClient(create_app())
    response = client.post("/forecast", json={"grid_id": "NOT_PECO", "horizon_hours": 24})
    assert response.status_code in {400, 503}
