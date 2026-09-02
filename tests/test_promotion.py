from smart_grid.models.promotion import should_promote


PROD = {
    "refit_metrics": {"test": {"wape": 0.03, "peak_mae": 300.0}},
}


def test_promotes_when_wape_improves():
    cand = {"refit_metrics": {"test": {"wape": 0.029, "peak_mae": 310.0}}}
    ok, _ = should_promote(cand, PROD, 0.002)
    assert ok


def test_holds_when_worse():
    cand = {"refit_metrics": {"test": {"wape": 0.04, "peak_mae": 200.0}}}
    ok, _ = should_promote(cand, PROD, 0.002)
    assert not ok


def test_peak_mae_breaks_wape_tie():
    cand = {"refit_metrics": {"test": {"wape": 0.0305, "peak_mae": 250.0}}}
    ok, _ = should_promote(cand, PROD, 0.002)
    assert ok
