# tests/unit/test_era5.py
from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion.era5 import (
    ERA5_VARIABLES,
    Era5PartialDownload,
    build_thailand_grid,
    fetch_era5_grid,
)


def test_era5_variables_has_required_fields():
    assert "precipitation" in ERA5_VARIABLES
    assert "temperature_2m" in ERA5_VARIABLES
    assert "relativehumidity_2m" in ERA5_VARIABLES


def test_era5_variables_count():
    assert len(ERA5_VARIABLES) == 7


def test_build_thailand_grid_covers_bounding_box():
    lats, lons = build_thailand_grid(spacing_deg=1.0)
    assert min(lats) >= 5.5
    assert max(lats) <= 20.5
    assert min(lons) >= 97.5
    assert max(lons) <= 105.7
    assert len(lats) == len(lons)


def test_build_thailand_grid_finer_spacing_has_more_points():
    lats_1, _ = build_thailand_grid(spacing_deg=1.0)
    lats_05, _ = build_thailand_grid(spacing_deg=0.5)
    assert len(lats_05) > len(lats_1)


def _make_mock_batch_df(lat=15.0, lon=102.0, n_hours=24):
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC"),
        "lat": lat,
        "lon": lon,
        "precipitation": [0.0] * n_hours,
        "temperature_2m": [28.0] * n_hours,
    })


def test_fetch_era5_grid_returns_dataframe_no_checkpoint():
    """Without checkpointing (test/small-data path) a DataFrame is returned directly."""
    mock_df = _make_mock_batch_df()
    with patch("src.ingestion.era5._fetch_batch", return_value=mock_df):
        result = fetch_era5_grid(
            [15.0], [102.0], "2023-01-01", "2023-01-02",
            checkpoint_dir=None,  # in-memory path — no Era5PartialDownload raised
        )
    assert isinstance(result, pd.DataFrame)
    assert "precipitation" in result.columns
    assert "lat" in result.columns


def test_fetch_era5_grid_all_batches_done_raises_is_complete(tmp_path):
    """When all checkpoint batches are done, Era5PartialDownload(is_complete=True) is raised.

    The caller (download_data.py) catches this and assembles the final parquet via
    PyArrow streaming — avoiding the OOM that pd.concat of ~140M rows would cause.
    """
    mock_df = _make_mock_batch_df()
    with patch("src.ingestion.era5._fetch_batch", return_value=mock_df):
        with pytest.raises(Era5PartialDownload) as exc_info:
            fetch_era5_grid(
                [15.0], [102.0], "2023-01-01", "2023-01-02",
                checkpoint_dir=tmp_path,
            )
    assert exc_info.value.is_complete is True


def test_fetch_era5_grid_deduplicates_no_checkpoint():
    """In-memory (no-checkpoint) path must not return duplicate rows."""
    mock_df = _make_mock_batch_df()
    with patch("src.ingestion.era5._fetch_batch", return_value=mock_df):
        result = fetch_era5_grid([15.0], [102.0], "2023-01-01", "2023-01-02",
                                 checkpoint_dir=None)
    assert result.duplicated(subset=["lat", "lon", "timestamp"]).sum() == 0


def test_fetch_era5_grid_resumes_from_checkpoint(tmp_path):
    """Second call skips already-completed batches (no extra _fetch_batch calls)."""
    mock_df = _make_mock_batch_df()
    call_count = {"n": 0}

    def counting_fetch(*args, **kwargs):
        call_count["n"] += 1
        return mock_df

    # First run — downloads the one batch; raises Era5PartialDownload(is_complete=True)
    with pytest.raises(Era5PartialDownload) as exc_info:
        with patch("src.ingestion.era5._fetch_batch", side_effect=counting_fetch):
            fetch_era5_grid([15.0], [102.0], "2023-01-01", "2023-01-02",
                            checkpoint_dir=tmp_path)
    assert exc_info.value.is_complete is True
    first_calls = call_count["n"]
    assert first_calls == 1

    # Second run — checkpoint already exists; no new _fetch_batch calls
    with pytest.raises(Era5PartialDownload) as exc_info2:
        with patch("src.ingestion.era5._fetch_batch", side_effect=counting_fetch):
            fetch_era5_grid([15.0], [102.0], "2023-01-01", "2023-01-02",
                            checkpoint_dir=tmp_path)
    assert exc_info2.value.is_complete is True
    second_calls = call_count["n"] - first_calls
    assert second_calls == 0  # checkpoint was reused — no fetch needed
