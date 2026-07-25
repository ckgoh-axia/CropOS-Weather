import pandas as pd
from unittest.mock import patch
from src.ingestion.nwp_baseline import fetch_nwp_at_point

def test_fetch_nwp_returns_forecast_dataframe():
    mock_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC"),
        "lat": [15.0] * 48, "lon": [102.0] * 48,
        "nwp_precip_mm": [0.0] * 48,
        "nwp_model": ["gfs_seamless"] * 48,
    })
    with patch("src.ingestion.nwp_baseline._fetch_forecast_run", return_value=mock_df):
        result = fetch_nwp_at_point(15.0, 102.0, "2023-01-01", "2023-01-03")
    assert "nwp_precip_mm" in result.columns
    assert "nwp_model" in result.columns
    assert len(result) > 0
