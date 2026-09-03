from smart_grid.serving.dispatch import ServingError, call


def test_health_via_dispatch():
    body = call("GET", "/health")
    assert body["grid_id"] == "PJM_PE"
    assert "model_loaded" in body


def test_forecast_rejects_other_grid_via_dispatch():
    try:
        call("POST", "/forecast", body={"grid_id": "NOT_PECO", "horizon_hours": 24})
    except ServingError as exc:
        assert exc.status in {400, 503}
        return
    raise AssertionError("expected ServingError")
