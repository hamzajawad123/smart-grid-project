from smart_grid.ingestion.eia_demand import _parse_period


def test_parse_eia_hour():
    ts = _parse_period("2022-01-01T00")
    assert str(ts.tz) == "UTC"
    assert ts.hour == 0
