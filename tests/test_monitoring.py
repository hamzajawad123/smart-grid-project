from smart_grid.serving.monitoring import _psi
import pandas as pd


def test_psi_identical_is_near_zero():
    s = pd.Series([1.0, 2.0, 3.0, 4.0] * 20)
    assert _psi(s, s) < 0.05


def test_psi_shifted_is_positive():
    a = pd.Series([1.0, 2.0, 3.0] * 30)
    b = pd.Series([8.0, 9.0, 10.0] * 30)
    assert _psi(a, b) > 0.2
