"""ERA5-Land ingestion via Open-Meteo archive API."""
from __future__ import annotations
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

ERA5_VARIABLES = [
    "temperature_2m", "dewpoint_2m", "relativehumidity_2m",
    "precipitation", "windspeed_10m", "winddirection_10m",
    "surface_pressure",
]
ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"


def _build_client() -> openmeteo_requests.Client:
    session = requests_cache.CachedSession(".cache/era5", expire_after=-1)
    session = retry(session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


def _fetch_single_point(
    client: openmeteo_requests.Client,
    lat: float, lon: float,
    start_date: str, end_date: str,
) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ERA5_VARIABLES, "timezone": "UTC",
    }
    response = client.weather_api(ERA5_URL, params=params)[0]
    hourly = response.Hourly()
    timestamps = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    data = {"timestamp": timestamps, "lat": lat, "lon": lon}
    for i, var in enumerate(ERA5_VARIABLES):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def fetch_era5_grid(
    lat_points: List[float],
    lon_points: List[float],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch ERA5-Land for a list of (lat, lon) grid points."""
    client = _build_client()
    frames = []
    for lat, lon in zip(lat_points, lon_points):
        logger.info(f"ERA5 ({lat:.2f}, {lon:.2f})")
        try:
            frames.append(_fetch_single_point(client, lat, lon, start_date, end_date))
        except Exception as exc:
            logger.warning(f"ERA5 failed ({lat},{lon}): {exc}")
    return pd.concat(frames, ignore_index=True)


def build_thailand_grid(spacing_deg: float = 0.25) -> tuple[List[float], List[float]]:
    """Generate regular grid points covering Thailand bounding box."""
    lats = list(np.arange(5.5, 20.5, spacing_deg))
    lons = list(np.arange(97.5, 105.7, spacing_deg))
    return [lat for lat in lats for _ in lons], [lon for _ in lats for lon in lons]
