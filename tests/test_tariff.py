from smart_grid.tariff.tou import period_for_ts, rate_usd_per_kwh, recommend
from smart_grid.serving.bands import stress_band


def test_weekday_afternoon_is_peak():
    # 2024-07-03 18:00 UTC = 14:00 EDT Wednesday
    assert period_for_ts("2024-07-03T18:00:00Z") == "peak"
    assert rate_usd_per_kwh("peak") == 0.33669


def test_weekday_6pm_is_off_peak():
    # 2024-07-03 22:00 UTC = 18:00 EDT — peak ends at 6pm
    assert period_for_ts("2024-07-03T22:00:00Z") == "off_peak"


def test_overnight_is_super_off_peak():
    # 2024-07-03 06:00 UTC = 02:00 EDT
    assert period_for_ts("2024-07-03T06:00:00Z") == "super_off_peak"
    assert rate_usd_per_kwh("super_off_peak") == 0.05556


def test_weekend_afternoon_is_off_peak():
    assert period_for_ts("2024-07-06T18:00:00Z") == "off_peak"


def test_memorial_day_afternoon_is_not_peak():
    # 2024-05-27 18:00 UTC = 14:00 EDT Memorial Day Monday
    assert period_for_ts("2024-05-27T18:00:00Z") == "off_peak"


def test_stress_bands():
    assert stress_band(1000, 4000, 5710, 6280) == "normal"
    assert stress_band(5000, 4000, 5710, 6280) == "elevated"
    assert stress_band(5710, 4000, 5710, 6280) == "high"
    assert stress_band(7000, 4000, 5710, 6280) == "critical"


def test_recommend_shifts_to_super_off_peak():
    hours = [
        {"ts_utc": "2024-07-03T18:00:00+00:00", "demand_mw": 6000, "band": "critical"},
        {"ts_utc": "2024-07-04T06:00:00+00:00", "demand_mw": 3000, "band": "normal"},
    ]
    rec = recommend(hours)
    assert rec["stress_hours_utc"]
    assert rec["suggested_off_peak_utc"]
    assert rec["policy"] == "peco_rate_r_tou"
    assert rec["rates_usd_per_kwh"]["peak"] > rec["rates_usd_per_kwh"]["off_peak"]
    assert rec["rates_usd_per_kwh"]["off_peak"] > rec["rates_usd_per_kwh"]["super_off_peak"]
