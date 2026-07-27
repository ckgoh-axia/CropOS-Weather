"""METAR ingestion from Iowa State University ASOS network."""
from __future__ import annotations

import logging
import time
from typing import List

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

IOWA_STATE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Timeout per yearly chunk. Iowa State returns ~8,760 rows/station-year (~500KB);
# 300s is generous even on slow GitHub Actions runners.
REQUEST_TIMEOUT_S = 300.0

# Delay between stations to be polite to Iowa State ASOS
STATION_DELAY_S = 2.0

# Delay between yearly requests within a single station.
# Iowa State 429s on the second request if fired within ~1s of the first.
# 5s is conservative and keeps total METAR download under 20 minutes.
INTER_YEAR_DELAY_S = 5.0

THAI_METAR_STATIONS: List[str] = [
    "VTUU", "VTUD", "VTUK", "VTUB", "VTUN", "VTUL",
    "VTCC", "VTCP", "VTCN", "VTBS", "VTBD", "VTBP",
    "VTSS", "VTSP", "VTSH", "VTSG",
]

STATION_COORDS: dict[str, tuple[float, float]] = {
    "VTUU": (15.25, 104.87), "VTUD": (17.39, 102.79),
    "VTUK": (16.47, 102.78), "VTUB": (15.23, 103.25),
    "VTUN": (14.95, 102.08), "VTUL": (17.44, 101.72),
    "VTCC": (18.77, 98.96),  "VTCP": (16.78, 100.15),
    "VTCN": (18.81, 100.78), "VTBS": (13.69, 100.75),
    "VTBD": (13.91, 100.61), "VTBP": (14.08, 101.70),
    "VTSS": (6.93, 100.43),  "VTSP": (8.11, 98.30),
    "VTSH": (9.13, 99.14),   "VTSG": (8.09, 98.99),
}


def _fetch_station_year(
    station: str,
    year: int,
    client: httpx.Client,
) -> pd.DataFrame:
    """Fetch one calendar year of METAR data for a single station."""
    params = {
        "station": station,
        "data": "all",
        "year1": str(year),   "month1": "1",  "day1": "1",
        "year2": str(year),   "month2": "12", "day2": "31",
        "tz": "Etc/UTC",
        "format": "comma",
        "latlon": "yes",
        "direct": "yes",
    }
    resp = client.get(IOWA_STATE_URL, params=params)
    resp.raise_for_status()
    return parse_metar_response(resp.text)


def fetch_metar_station(
    station: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch METAR for one station across the full date range.

    Splits into one HTTP call per calendar year so each request stays
    small (~500 KB) and never hits a timeout.
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    yearly_frames: List[pd.DataFrame] = []

    years = list(range(start_year, end_year + 1))
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        for idx, year in enumerate(years):
            try:
                df = _fetch_station_year(station, year, client)
                if len(df) > 0:
                    yearly_frames.append(df)
                    logger.info(f"METAR | {station} {year}: {len(df):,} rows")
                else:
                    logger.warning(f"METAR | {station} {year}: 0 rows returned")
            except Exception as exc:
                logger.warning(f"METAR | {station} {year} failed: {exc}")
            # Throttle between yearly requests to avoid Iowa State 429 rate limit.
            # Without this delay every request after the first 429s immediately.
            if idx < len(years) - 1:
                time.sleep(INTER_YEAR_DELAY_S)

    if not yearly_frames:
        return pd.DataFrame(columns=[
            "station", "timestamp", "precip_mm", "rain_event",
            "tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby",
        ])
    return pd.concat(yearly_frames, ignore_index=True)


def parse_metar_response(csv_text: str) -> pd.DataFrame:
    """Parse Iowa State ASOS CSV into a clean DataFrame."""
    from io import StringIO
    lines = [ln for ln in csv_text.splitlines() if not ln.startswith("#") and ln.strip()]
    if len(lines) <= 1:
        return pd.DataFrame()
    df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
    if df.empty:
        return df
    df = df.rename(columns={"valid": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["precip_mm"] = pd.to_numeric(df.get("p01i", 0), errors="coerce").fillna(0) * 25.4
    wxcodes = df.get("wxcodes", pd.Series([""] * len(df))).fillna("")
    df["rain_event"] = wxcodes.str.contains(r"\bRA\b|\bTS\b|\bSH\b", regex=True)
    df["station"] = df["station"].str.strip()
    # Iowa State ASOS uses "M" for missing values; coerce to float so PyArrow
    # can serialize the parquet without ArrowTypeError on object columns.
    for col in ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["station", "timestamp", "precip_mm", "rain_event",
            "tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def fetch_all_thai_stations(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch METAR for all 16 Thai stations and concatenate into one DataFrame."""
    frames = []
    for i, station in enumerate(THAI_METAR_STATIONS):
        logger.info(f"METAR | station {i+1}/{len(THAI_METAR_STATIONS)}: {station}")
        df = fetch_metar_station(station, start_date, end_date)
        if len(df) > 0:
            lat, lon = STATION_COORDS[station]
            df["lat"] = lat
            df["lon"] = lon
            frames.append(df)
        else:
            logger.warning(f"METAR | {station}: no data — skipping")
        if i < len(THAI_METAR_STATIONS) - 1:
            time.sleep(STATION_DELAY_S)

    if not frames:
        logger.error("METAR | all 16 stations returned no data")
        return pd.DataFrame(columns=[
            "station", "timestamp", "precip_mm", "rain_event",
            "tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby", "lat", "lon",
        ])
    result = pd.concat(frames, ignore_index=True)
    logger.info(
        f"METAR | total: {len(result):,} rows across "
        f"{result['station'].nunique()} stations"
    )
    return result
