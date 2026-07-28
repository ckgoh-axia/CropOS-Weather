"""Unit tests for NWP baseline ingestion (nwp_baseline.py).

All tests mock _build_client so no real HTTP calls are made.
The Open-Meteo response object has this shape:
    responses[0].Hourly() -> hourly
    hourly.Time()          -> int (unix seconds, start)
    hourly.TimeEnd()       -> int (unix seconds, end)
    hourly.Interval()      -> int (seconds per step, e.g. 3600)
    hourly.VariablesLength()  -> int
    hourly.Variables(i).ValuesAsNumpy() -> np.ndarray
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingestion.nwp_baseline import (
    NWP_DEFAULT_VARIABLES,
    fetch_all_stations,
    fetch_nwp_at_point,
    fetch_nwp_at_station,
)

# 2023-01-01 00:00 UTC
_START_UNIX = 1_672_531_200


def _mock_client(variables: list[str], n_hours: int = 24) -> MagicMock:
    """Return a fake openmeteo Client whose weather_api() mimics a real response."""
    mock_var = MagicMock()
    mock_var.ValuesAsNumpy.return_value = np.zeros(n_hours, dtype=np.float32)

    hourly = MagicMock()
    hourly.Time.return_value = _START_UNIX
    hourly.TimeEnd.return_value = _START_UNIX + n_hours * 3600
    hourly.Interval.return_value = 3600
    hourly.VariablesLength.return_value = len(variables)
    hourly.Variables.return_value = mock_var

    response = MagicMock()
    response.Hourly.return_value = hourly

    client = MagicMock()
    client.weather_api.return_value = [response]
    return client


# ---------------------------------------------------------------------------
# fetch_nwp_at_station
# ---------------------------------------------------------------------------

def test_fetch_nwp_at_station_returns_dataframe():
    variables = ["precipitation", "temperature_2m"]
    client = _mock_client(variables, n_hours=24)

    with patch("src.ingestion.nwp_baseline._build_client", return_value=client):
        df = fetch_nwp_at_station(
            "VTBS", 13.69, 100.75, "2023-01-01", "2023-01-01", variables
        )

    assert isinstance(df, pd.DataFrame)
    assert set(["station", "timestamp", "lat", "lon"]).issubset(df.columns)
    assert "nwp_precipitation" in df.columns
    assert "nwp_temperature_2m" in df.columns
    assert len(df) == 24
    assert (df["station"] == "VTBS").all()


def test_fetch_nwp_at_station_uses_default_variables():
    client = _mock_client(NWP_DEFAULT_VARIABLES, n_hours=24)

    with patch("src.ingestion.nwp_baseline._build_client", return_value=client):
        df = fetch_nwp_at_station("VTBS", 13.69, 100.75, "2023-01-01", "2023-01-01")

    nwp_cols = [c for c in df.columns if c.startswith("nwp_")]
    assert len(nwp_cols) == len(NWP_DEFAULT_VARIABLES)


def test_fetch_nwp_at_station_coerces_nwp_columns_to_float():
    variables = ["precipitation"]
    client = _mock_client(variables, n_hours=4)

    with patch("src.ingestion.nwp_baseline._build_client", return_value=client):
        df = fetch_nwp_at_station(
            "VTBS", 13.69, 100.75, "2023-01-01", "2023-01-01", variables
        )

    assert np.issubdtype(df["nwp_precipitation"].dtype, np.floating)


def test_fetch_nwp_at_station_warns_on_variable_count_mismatch(caplog):
    # Request 2 variables but API only returns 1.
    requested = ["precipitation", "temperature_2m"]
    client = _mock_client(["precipitation"], n_hours=4)  # VariablesLength=1

    with patch("src.ingestion.nwp_baseline._build_client", return_value=client):
        with caplog.at_level(logging.WARNING, logger="src.ingestion.nwp_baseline"):
            fetch_nwp_at_station(
                "VTBS", 13.69, 100.75, "2023-01-01", "2023-01-01", requested
            )

    assert "requested 2 variables" in caplog.text
    assert "API returned 1" in caplog.text


def test_fetch_nwp_at_station_raises_on_api_failure():
    client = MagicMock()
    client.weather_api.side_effect = RuntimeError("connection timeout")

    with patch("src.ingestion.nwp_baseline._build_client", return_value=client):
        with pytest.raises(RuntimeError, match="connection timeout"):
            fetch_nwp_at_station("VTBS", 13.69, 100.75, "2023-01-01", "2023-01-01")


def test_fetch_nwp_at_station_timestamps_are_utc():
    variables = ["precipitation"]
    client = _mock_client(variables, n_hours=6)

    with patch("src.ingestion.nwp_baseline._build_client", return_value=client):
        df = fetch_nwp_at_station(
            "VTBS", 13.69, 100.75, "2023-01-01", "2023-01-01", variables
        )

    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


# ---------------------------------------------------------------------------
# fetch_all_stations
# ---------------------------------------------------------------------------

def _station_df(station: str, n: int = 4) -> pd.DataFrame:
    lat, lon = {"VTBS": (13.69, 100.75), "VTBD": (13.91, 100.61)}.get(
        station, (15.0, 102.0)
    )
    return pd.DataFrame({
        "station": [station] * n,
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "lat": [lat] * n,
        "lon": [lon] * n,
        "nwp_precipitation": [0.0] * n,
    })


def test_fetch_all_stations_restores_from_checkpoint(tmp_path):
    """Station with an existing checkpoint parquet is not re-fetched."""
    station_coords = {"VTBS": (13.69, 100.75)}
    _station_df("VTBS").to_parquet(tmp_path / "VTBS.parquet", index=False)

    with patch("src.ingestion.nwp_baseline.fetch_nwp_at_station") as mock_fetch:
        result = fetch_all_stations(
            station_coords, "2023-01-01", "2023-01-01", checkpoint_dir=tmp_path
        )

    mock_fetch.assert_not_called()
    assert len(result) == 4


def test_fetch_all_stations_saves_checkpoint_after_fetch(tmp_path):
    """A freshly-fetched station is written to the checkpoint dir."""
    station_coords = {"VTBS": (13.69, 100.75)}

    with patch(
        "src.ingestion.nwp_baseline.fetch_nwp_at_station",
        return_value=_station_df("VTBS"),
    ):
        fetch_all_stations(
            station_coords, "2023-01-01", "2023-01-01", checkpoint_dir=tmp_path
        )

    assert (tmp_path / "VTBS.parquet").exists()


def test_fetch_all_stations_skips_failed_station(tmp_path):
    """A failing station is logged and excluded; successful stations still return."""
    station_coords = {"VTBS": (13.69, 100.75), "VTBD": (13.91, 100.61)}

    def _side_effect(station, *args, **kwargs):
        if station == "VTBS":
            raise RuntimeError("timeout")
        return _station_df("VTBD")

    with patch("src.ingestion.nwp_baseline.fetch_nwp_at_station", side_effect=_side_effect):
        with patch("src.ingestion.nwp_baseline.time") as mock_time:
            mock_time.sleep = MagicMock()
            result = fetch_all_stations(
                station_coords, "2023-01-01", "2023-01-01", checkpoint_dir=tmp_path
            )

    assert result["station"].nunique() == 1
    assert result["station"].iloc[0] == "VTBD"


def test_fetch_all_stations_raises_if_all_stations_fail(tmp_path):
    station_coords = {"VTBS": (13.69, 100.75)}

    with patch(
        "src.ingestion.nwp_baseline.fetch_nwp_at_station",
        side_effect=RuntimeError("fail"),
    ):
        with pytest.raises(RuntimeError, match="all stations failed"):
            fetch_all_stations(
                station_coords, "2023-01-01", "2023-01-01", checkpoint_dir=tmp_path
            )


def test_fetch_all_stations_works_without_checkpoint_dir():
    """checkpoint_dir=None is valid — no disk I/O attempted."""
    station_coords = {"VTBS": (13.69, 100.75)}

    with patch(
        "src.ingestion.nwp_baseline.fetch_nwp_at_station",
        return_value=_station_df("VTBS"),
    ):
        result = fetch_all_stations(
            station_coords, "2023-01-01", "2023-01-01", checkpoint_dir=None
        )

    assert len(result) == 4


def test_fetch_all_stations_deduplicates_rows():
    """Duplicate (station, timestamp) rows are dropped in the final concat."""
    station_coords = {"VTBS": (13.69, 100.75)}
    # Return a df with a duplicate row
    df = _station_df("VTBS", n=4)
    df_dup = pd.concat([df, df.iloc[:1]], ignore_index=True)  # 5 rows, 1 duplicate

    with patch(
        "src.ingestion.nwp_baseline.fetch_nwp_at_station", return_value=df_dup
    ):
        result = fetch_all_stations(
            station_coords, "2023-01-01", "2023-01-01", checkpoint_dir=None
        )

    assert len(result) == 4  # duplicate dropped


# ---------------------------------------------------------------------------
# fetch_nwp_at_point — legacy wrapper
# ---------------------------------------------------------------------------

def test_fetch_nwp_at_point_renames_column_to_nwp_precip_mm():
    inner_df = pd.DataFrame({
        "station": ["_point"] * 4,
        "timestamp": pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC"),
        "lat": [15.0] * 4,
        "lon": [102.0] * 4,
        "nwp_precipitation": [1.0, 0.0, 2.0, 0.5],
    })

    with patch(
        "src.ingestion.nwp_baseline.fetch_nwp_at_station", return_value=inner_df
    ):
        result = fetch_nwp_at_point(15.0, 102.0, "2023-01-01", "2023-01-01")

    assert "nwp_precip_mm" in result.columns
    assert "nwp_precipitation" not in result.columns


# ---------------------------------------------------------------------------
# NWP_DEFAULT_VARIABLES contract
# ---------------------------------------------------------------------------

def test_nwp_default_variables_contains_convective_vars():
    assert "cape" in NWP_DEFAULT_VARIABLES
    assert "lifted_index" in NWP_DEFAULT_VARIABLES


def test_nwp_default_variables_excludes_invalid_api_vars():
    # precipitable_water is rejected by gfs_seamless historical-forecast-api;
    # it must stay out of the default list until a valid name is confirmed.
    assert "precipitable_water" not in NWP_DEFAULT_VARIABLES


def test_nwp_default_variables_has_cloud_layers():
    assert "cloudcover_low" in NWP_DEFAULT_VARIABLES
    assert "cloudcover_mid" in NWP_DEFAULT_VARIABLES
    assert "cloudcover_high" in NWP_DEFAULT_VARIABLES


def test_nwp_default_variables_has_multi_level_winds():
    assert "windspeed_850hPa" in NWP_DEFAULT_VARIABLES
    assert "windspeed_700hPa" in NWP_DEFAULT_VARIABLES
    assert "windspeed_200hPa" in NWP_DEFAULT_VARIABLES


def test_nwp_default_variables_has_soil_moisture():
    assert "soil_moisture_0_to_7cm" in NWP_DEFAULT_VARIABLES


def test_nwp_default_variables_no_duplicates():
    assert len(NWP_DEFAULT_VARIABLES) == len(set(NWP_DEFAULT_VARIABLES))
