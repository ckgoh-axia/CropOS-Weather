"""Open-Meteo historical forecast ingestion — GFS NWP baseline."""
from __future__ import annotations
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import logging

logger = logging.getLogger(__name__)

NWP_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
NWP_MODEL = "gfs_seamless"


def _build_client():
    session = requests_cache.CachedSession(".cache/nwp", expire_after=-1)
    return retry(session, retries=5, backoff_factor=0.2)


def _fetch_forecast_run(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    client = openmeteo_requests.Client(session=_build_client())
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["precipitation"],
        "models": NWP_MODEL,
        "timezone": "UTC",
    }
    response = client.weather_api(NWP_URL, params=params)[0]
    hourly = response.Hourly()
    timestamps = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    return pd.DataFrame({
        "timestamp": timestamps, "lat": lat, "lon": lon,
        "nwp_precip_mm": hourly.Variables(0).ValuesAsNumpy(),
        "nwp_model": NWP_MODEL,
    })


def fetch_nwp_at_point(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_forecast_run(lat, lon, start_date, end_date)
