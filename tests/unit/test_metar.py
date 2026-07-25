# tests/unit/test_metar.py
import pandas as pd
import pytest
import respx
import httpx
from src.ingestion.metar import fetch_metar_station, parse_metar_response

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
    assert df.iloc[1]["rain_event"] == True
    assert df.iloc[0]["rain_event"] == False

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
