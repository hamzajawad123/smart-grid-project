from smart_grid.config import load_params


def test_params_has_peco_subba():
    params = load_params()
    assert params["grid_id"] == "PJM_PE"
    assert params["eia"]["parent"] == "PJM"
    assert params["eia"]["subba"] == "PE"
    assert params["weather"]["latitude"] == 39.95
