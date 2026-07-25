"""ERA5-Land ingestion via Open-Meteo archive API."""
from __future__ import annotations
import time
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
BATCH_SIZE = 50       # points per API call — Open-Meteo accepts arrays of lat/lon
BATCH_DELAY_S = 2.0   # seconds between batches — stays well under free tier limits


def _build_client() -> openmeteo_requests.Client:
    session = requests_cache.CachedSession(".cache/era5", expire_after=-1)
    session = retry(session, retries=5, backoff_factor=1.0)  # 1s, 2s, 4s, 8s, 16s
    return openmeteo_requests.Client(session=session)


def _fetch_batch(
    client: openmeteo_requests.Client,
    lats: List[float],
    lons: List[float],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch ERA5-Land for a batch of points in one API call."""
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ERA5_VARIABLES,
        "timezone": "UTC",
    }
    responses = client.weather_api(ERA5_URL, params=params)
    frames = []
    for r, lat, lon in zip(responses, lats, lons):
        hourly = r.Hourly()
        timestamps = pd.date_range(
            start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
            end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        data = {"timestamp": timestamps, "lat": lat, "lon": lon}
        for i, var in enumerate(ERA5_VARIABLES):
            data[var] = hourly.Variables(i).ValuesAsNumpy()
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def fetch_era5_grid(
    lat_points: List[float],
    lon_points: List[float],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch ERA5-Land for all grid points using batched API calls.

    Batches BATCH_SIZE points per request to minimise API calls and
    avoid minutely rate limits on the Open-Meteo free tier.
    1,980 Thai grid points → 40 batched calls instead of 1,980 single calls.
    """
    client = _build_client()
    all_frames = []
    pairs = list(zip(lat_points, lon_points))
    n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i: i + BATCH_SIZE]
        b_lats = [p[0] for p in batch]
        b_lons = [p[1] for p in batch]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"ERA5 | batch {batch_num}/{n_batches} ({len(batch)} points)")
        try:
            all_frames.append(_fetch_batch(client, b_lats, b_lons, start_date, end_date))
        except Exception as exc:
            logger.warning(f"ERA5 | batch {batch_num} failed: {exc}")
        if i + BATCH_SIZE < len(pairs):
            time.sleep(BATCH_DELAY_S)

    return pd.concat(all_frames, ignore_index=True)


def build_thailand_grid(spacing_deg: float = 0.25) -> tuple[List[float], List[float]]:
    """Generate regular grid points covering Thailand bounding box."""
    lats = list(np.arange(5.5, 20.5, spacing_deg))
    lons = list(np.arange(97.5, 105.7, spacing_deg))
    return [lat for lat in lats for _ in lons], [lon for _ in lats for lon in lons]
