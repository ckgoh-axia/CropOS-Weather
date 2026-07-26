# tests/unit/test_metar.py
from unittest.mock import patch

import httpx
import pandas as pd
import respx

from src.ingestion.metar import (
    fetch_all_thai_stations,
    fetch_metar_station,
    parse_metar_response,
)

SAMPLE_CSV = """station,valid,tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,skyc1,wxcodes
VTUU,2023-01-01 00:00,75.2,68.0,78,180,8,0.00,29.92,1013.2,6.21,BKN,
VTUU,2023-01-01 01:00,74.3,67.5,80,175,6,0.05,29.91,1013.0,5.00,OVC,RA
"""

def test_parse_metar_returns_dataframe():
    df = parse_metar_response(SAMPLE_CSV)
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "precip_mm" in df.columns
    assert "rain_event" in df.columns
    assert len(df) == 2

def test_parse_metar_detects_rain_code():
    df = parse_metar_response(SAMPLE_CSV)
    assert df.iloc[1]["rain_event"]
    assert not df.iloc[0]["rain_event"]

def test_parse_metar_converts_inches_to_mm():
    df = parse_metar_response(SAMPLE_CSV)
    # 0.05 inches × 25.4 = 1.27 mm
    assert abs(df.iloc[1]["precip_mm"] - 1.27) < 0.01

@respx.mock
def test_fetch_metar_station_calls_iowa_state():
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    respx.get(url).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    df = fetch_metar_station("VTUU", "2023-01-01", "2023-01-02")
    assert len(df) == 2
    assert df.iloc[0]["station"] == "VTUU"


def test_parse_metar_empty_response_returns_empty_df():
    """Header-only or blank response should return an empty DataFrame."""
    empty = "station,valid,tmpf\n"
    df = parse_metar_response(empty)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_parse_metar_comment_lines_ignored():
    csv = "# This is a comment\n" + SAMPLE_CSV
    df = parse_metar_response(csv)
    assert len(df) == 2  # comment stripped, same rows as SAMPLE_CSV


def test_fetch_all_thai_stations_empty_when_all_fail():
    """All stations fail → returns empty DataFrame with correct columns."""
    with patch("src.ingestion.metar.fetch_metar_station", return_value=pd.DataFrame()):
        result = fetch_all_thai_stations("2023-01-01", "2023-01-02")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "station" in result.columns


def test_fetch_all_thai_stations_concatenates_results():
    """Two stations each returning 2 rows → 4 combined rows."""
    mock_df = pd.DataFrame({
        "station": ["VTUU", "VTUU"],
        "timestamp": pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC"),
        "precip_mm": [0.0, 1.27],
        "rain_event": [False, True],
    })
    with patch("src.ingestion.metar.fetch_metar_station", return_value=mock_df), \
         patch("src.ingestion.metar.time.sleep"):
        result = fetch_all_thai_stations("2023-01-01", "2023-01-02")
    assert len(result) == 2 * 16  # 16 stations × 2 rows
    assert "lat" in result.columns
    assert "lon" in result.columns
