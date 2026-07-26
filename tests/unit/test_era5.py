from unittest.mock import patch

import pandas as pd

from src.ingestion.era5 import ERA5_VARIABLES, build_thailand_grid, fetch_era5_grid


def test_era5_variables_has_required_fields():
    assert "precipitation" in ERA5_VARIABLES
    assert "temperature_2m" in ERA5_VARIABLES
    assert "relativehumidity_2m" in ERA5_VARIABLES

def test_build_thailand_grid_covers_bounding_box():
    lats, lons = build_thailand_grid(spacing_deg=1.0)
    assert min(lats) >= 5.5
    assert max(lats) <= 20.5
    assert min(lons) >= 97.5
    assert max(lons) <= 105.7
    assert len(lats) == len(lons)

def test_fetch_era5_grid_returns_dataframe():
    mock_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC"),
        "lat": [15.0] * 24, "lon": [102.0] * 24,
        "precipitation": [0.0] * 24, "temperature_2m": [28.0] * 24,
    })
    with patch("src.ingestion.era5._fetch_single_point", return_value=mock_df):
        result = fetch_era5_grid([15.0], [102.0], "2023-01-01", "2023-01-02")
    assert isinstance(result, pd.DataFrame)
    assert "precipitation" in result.columns
    assert "lat" in result.columns
