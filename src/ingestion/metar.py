"""METAR ingestion from Iowa State University ASOS network."""
from __future__ import annotations
import httpx
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

IOWA_STATE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

THAI_METAR_STATIONS: List[str] = [
    "VTUU", "VTUD", "VTUK", "VTUB", "VTUN", "VTUL",
    "VTCC", "VTCP", "VTCN", "VTBS", "VTBD", "VTBP",
    "VTSS", "VTSP", "VTSH", "VTSG",
]

# Official coordinates for each Thai METAR station (lat, lon)
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


def fetch_metar_station(
    station: str,
    start_date: str,
    end_date: str,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch hourly METAR data for one ICAO station from Iowa State ASOS archive."""
    params = {
        "station": station,
        "data": "all",
        "year1": start_date[:4],   "month1": start_date[5:7],  "day1": start_date[8:10],
        "year2": end_date[:4],     "month2": end_date[5:7],    "day2": end_date[8:10],
        "tz": "Etc/UTC",
        "format": "comma",
        "latlon": "yes",
        "direct": "yes",
        "report_type": "1",  # routine hourly only
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(IOWA_STATE_URL, params=params)
        resp.raise_for_status()
    return parse_metar_response(resp.text)


def parse_metar_response(csv_text: str) -> pd.DataFrame:
    """Parse Iowa State ASOS CSV into a clean DataFrame."""
    from io import StringIO
    lines = [ln for ln in csv_text.splitlines() if not ln.startswith("#") and ln.strip()]
    df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
    df = df.rename(columns={"valid": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["precip_mm"] = pd.to_numeric(df.get("p01i", 0), errors="coerce").fillna(0) * 25.4
    wxcodes = df.get("wxcodes", pd.Series([""] * len(df))).fillna("")
    df["rain_event"] = wxcodes.str.contains(r"\bRA\b|\bTS\b|\bSH\b", regex=True)
    df["station"] = df["station"].str.strip()
    keep = ["station", "timestamp", "precip_mm", "rain_event",
            "tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def fetch_all_thai_stations(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch METAR for all 16 Thai stations and concatenate into one DataFrame."""
    frames = []
    for station in THAI_METAR_STATIONS:
        logger.info(f"Fetching METAR: {station} {start_date}→{end_date}")
        try:
            df = fetch_metar_station(station, start_date, end_date)
            lat, lon = STATION_COORDS[station]
            df["lat"] = lat
            df["lon"] = lon
            frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed {station}: {exc}")
    return pd.concat(frames, ignore_index=True)
