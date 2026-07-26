# tests/unit/test_labels.py
"""Tests for src/preprocessing/labels.py — METAR/ERA5 label merging."""
import pandas as pd
import pytest

from src.preprocessing.labels import (
    _extract_era5_labels,
    _extract_metar_labels,
    build_labels,
)


def _ts(s):
    return pd.to_datetime(s, utc=True)


# ── _extract_metar_labels ──────────────────────────────────────────────────────

def _metar_df():
    """Matches the output format of fetch_all_thai_stations() / parse_metar_response().

    parse_metar_response() already converts p01i (inches) → precip_mm and
    renames 'valid' → 'timestamp', so the labels module receives these columns.
    """
    return pd.DataFrame({
        "station": ["VTBS", "VTBS", "VTBS"],
        "timestamp": [_ts("2023-06-01 00:00"), _ts("2023-06-01 01:00"), _ts("2023-06-01 02:00")],
        "precip_mm": [0.0, 1.27, None],  # already in mm; None simulates missing gauge
        "rain_event": [False, True, False],
        "lat": [13.69, 13.69, 13.69],
        "lon": [100.75, 100.75, 100.75],
    })


def test_extract_metar_passthrough_mm():
    """precip_mm passes through unchanged (no second inches→mm conversion)."""
    result = _extract_metar_labels(_metar_df())
    assert abs(result.iloc[1]["precip_mm"] - 1.27) < 0.01


def test_extract_metar_drops_null_precip():
    result = _extract_metar_labels(_metar_df())
    assert len(result) == 2  # third row (None) dropped


def test_extract_metar_label_source():
    result = _extract_metar_labels(_metar_df())
    assert (result["label_source"] == "metar").all()


def test_extract_metar_has_required_columns():
    result = _extract_metar_labels(_metar_df())
    for col in ["lat", "lon", "timestamp", "precip_mm", "label_source"]:
        assert col in result.columns


def test_extract_metar_timestamp_column_present():
    result = _extract_metar_labels(_metar_df())
    assert "timestamp" in result.columns


# ── _extract_era5_labels ───────────────────────────────────────────────────────

def _era5_df():
    return pd.DataFrame({
        "lat": [15.0, 15.0],
        "lon": [102.0, 102.0],
        "timestamp": [_ts("2023-06-01 00:00"), _ts("2023-06-01 01:00")],
        "precipitation": [2.5, 0.0],
    })


def test_extract_era5_label_source():
    result = _extract_era5_labels(_era5_df())
    assert (result["label_source"] == "era5").all()


def test_extract_era5_has_required_columns():
    result = _extract_era5_labels(_era5_df())
    for col in ["lat", "lon", "timestamp", "precip_mm", "label_source"]:
        assert col in result.columns


def test_extract_era5_missing_precip_column_raises():
    df = pd.DataFrame({"lat": [15.0], "lon": [102.0], "timestamp": [_ts("2023-01-01")]})
    with pytest.raises(ValueError, match="No precipitation column"):
        _extract_era5_labels(df)


def test_extract_era5_uses_precipitation_sum_if_present():
    df = pd.DataFrame({
        "lat": [15.0], "lon": [102.0],
        "timestamp": [_ts("2023-01-01")],
        "precipitation_sum": [3.0],
    })
    result = _extract_era5_labels(df)
    assert result.iloc[0]["precip_mm"] == pytest.approx(3.0)


def test_extract_era5_converts_m_per_hour_if_tiny():
    """Values < 0.05 max trigger the m/h → mm/h conversion."""
    df = pd.DataFrame({
        "lat": [15.0] * 3, "lon": [102.0] * 3,
        "timestamp": [_ts("2023-01-01"), _ts("2023-01-02"), _ts("2023-01-03")],
        "precipitation": [0.001, 0.002, 0.003],  # looks like m/h
    })
    result = _extract_era5_labels(df)
    # Should be multiplied × 1000
    assert result["precip_mm"].max() > 1.0


# ── build_labels ───────────────────────────────────────────────────────────────

def test_build_labels_metar_takes_priority():
    """When a METAR row overlaps an ERA5 row in lat/lon/time, keep the METAR one."""
    ts = _ts("2023-06-01 00:00")
    metar = pd.DataFrame({
        "station": ["VTBS"],
        "timestamp": [ts],
        "precip_mm": [2.54],
        "lat": [13.69],
        "lon": [100.75],
    })
    era5 = pd.DataFrame({
        "lat": [13.69], "lon": [100.75],
        "timestamp": [ts],
        "precipitation": [5.0],  # would be 5mm if not shadowed
    })
    result = build_labels(metar, era5)
    metar_rows = result[result["label_source"] == "metar"]
    era5_rows_at_station = result[
        (result["label_source"] == "era5") &
        (result["lat"].round(2) == 13.69) &
        (result["lon"].round(2) == 100.75)
    ]
    assert len(metar_rows) == 1
    assert len(era5_rows_at_station) == 0


def test_build_labels_rain_flag_respects_threshold():
    ts = _ts("2023-06-01 00:00")
    metar = pd.DataFrame({
        "station": ["VTBS", "VTBS"],
        "timestamp": [ts, _ts("2023-06-01 01:00")],
        "precip_mm": [0.508, 0.0],  # 0.508mm < 1mm threshold, 0mm
        "lat": [13.69, 13.69],
        "lon": [100.75, 100.75],
    })
    era5 = pd.DataFrame({
        "lat": [15.0], "lon": [102.0],
        "timestamp": [ts],
        "precipitation": [0.0],
    })
    result = build_labels(metar, era5, precip_threshold_mm=1.0)
    # 0.508 mm < 1.0 mm threshold → not rain
    assert not result[result["label_source"] == "metar"].iloc[0]["rain"]


def test_build_labels_output_columns():
    ts = _ts("2023-06-01 00:00")
    metar = pd.DataFrame({
        "station": ["VTBS"], "timestamp": [ts],
        "precip_mm": [0.0], "lat": [13.69], "lon": [100.75],
    })
    era5 = pd.DataFrame({
        "lat": [15.0], "lon": [102.0],
        "timestamp": [ts], "precipitation": [1.0],
    })
    result = build_labels(metar, era5)
    for col in ["lat", "lon", "timestamp", "precip_mm", "label_source", "rain"]:
        assert col in result.columns
