"""Tests for temporal alignment and ERA5-METAR merging."""
import pandas as pd
import pytest
from src.preprocessing.align import align_to_hourly_utc, merge_era5_metar
from src.preprocessing.qc import flag_metar_outliers


def _ts(s):
    return pd.to_datetime(s, utc=True)


def test_align_to_hourly_sums_within_window():
    df = pd.DataFrame({
        "timestamp": [_ts("2023-01-01 00:15"), _ts("2023-01-01 00:45"), _ts("2023-01-01 01:15")],
        "lat": 15.0, "lon": 102.0,
        "precip_mm": [1.0, 2.0, 3.0],
    })
    result = align_to_hourly_utc(df, value_col="precip_mm", agg="sum")
    assert len(result) == 2
    assert result.iloc[0]["precip_mm"] == pytest.approx(3.0)


def test_merge_era5_metar_joins_on_timestamp():
    era5 = pd.DataFrame({
        "timestamp": [_ts("2023-01-01 00:00")],
        "lat": 15.0, "lon": 102.0,
        "precipitation": [0.5], "temperature_2m": [28.0],
    })
    metar = pd.DataFrame({
        "timestamp": [_ts("2023-01-01 00:00")],
        "lat": 15.25, "lon": 104.87,
        "precip_mm": [0.0], "rain_event": [False], "station": "VTUU",
    })
    result = merge_era5_metar(era5, metar)
    assert "precip_mm" in result.columns
    assert "temperature_2m" in result.columns


def test_flag_outliers_marks_impossible_precip():
    df = pd.DataFrame({"precip_mm": [0.0, 5.0, 999.0, -1.0]})
    result = flag_metar_outliers(df)
    assert result.iloc[2]["qc_flag"] == "outlier_high"
    assert result.iloc[3]["qc_flag"] == "outlier_low"
    assert result.iloc[0]["qc_flag"] == "ok"
