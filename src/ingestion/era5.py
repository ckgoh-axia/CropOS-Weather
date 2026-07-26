"""ERA5-Land ingestion via Open-Meteo archive API."""
from __future__ import annotations
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

ERA5_VARIABLES = [
    "temperature_2m", "dewpoint_2m", "relativehumidity_2m",
    "precipitation", "windspeed_10m", "winddirection_10m",
    "surface_pressure",
]
ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"

# archive-api.open-meteo.com is much stricter than the forecast endpoint.
# 10 points per call and 15s between calls keeps us safely within the free tier.
BATCH_SIZE = 10
BATCH_DELAY_S = 15.0


def _build_client() -> openmeteo_requests.Client:
    session = requests_cache.CachedSession(".cache/era5", expire_after=-1)
    session = retry(session, retries=5, backoff_factor=2.0)  # 2s, 4s, 8s, 16s, 32s
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
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch ERA5-Land for all grid points using batched API calls.

    Uses a checkpoint file so interrupted runs resume rather than restart.
    Each completed batch is appended to the checkpoint file immediately.

    Args:
        checkpoint_dir: Directory to store batch checkpoints. Defaults to
                        the system temp directory. Set to None to disable.
    """
    client = _build_client()
    pairs = list(zip(lat_points, lon_points))
    n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    # Checkpoint: track which batch indices are already done
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "era5_checkpoint.parquet"
    else:
        checkpoint_path = None

    completed_frames: List[pd.DataFrame] = []
    completed_batches: set[int] = set()

    if checkpoint_path and checkpoint_path.exists():
        prev = pd.read_parquet(checkpoint_path)
        completed_frames.append(prev)
        # Infer which pairs are already downloaded
        done_pairs = set(zip(prev["lat"].round(6), prev["lon"].round(6)))
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i: i + BATCH_SIZE]
            if all((round(p[0], 6), round(p[1], 6)) in done_pairs for p in batch):
                completed_batches.add(i // BATCH_SIZE)
        logger.info(
            f"ERA5 | resuming: {len(completed_batches)}/{n_batches} batches already done "
            f"({len(prev):,} rows in checkpoint)"
        )

    for i in range(0, len(pairs), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        if (i // BATCH_SIZE) in completed_batches:
            logger.info(f"ERA5 | batch {batch_num}/{n_batches} — skipped (checkpoint)")
            continue

        batch = pairs[i: i + BATCH_SIZE]
        b_lats = [p[0] for p in batch]
        b_lons = [p[1] for p in batch]
        logger.info(f"ERA5 | batch {batch_num}/{n_batches} ({len(batch)} points, "
                    f"lat {min(b_lats):.2f}–{max(b_lats):.2f})")
        try:
            frame = _fetch_batch(client, b_lats, b_lons, start_date, end_date)
            completed_frames.append(frame)
            # Write checkpoint after every successful batch
            if checkpoint_path:
                pd.concat(completed_frames, ignore_index=True).to_parquet(
                    checkpoint_path, index=False
                )
            logger.info(f"ERA5 | batch {batch_num}/{n_batches} done ({len(frame):,} rows)")
        except Exception as exc:
            logger.error(f"ERA5 | batch {batch_num}/{n_batches} FAILED: {exc}")
            logger.error("ERA5 | stopping — re-run the script to resume from checkpoint")
            # Return whatever we have so far rather than crashing
            break

        if i + BATCH_SIZE < len(pairs):
            time.sleep(BATCH_DELAY_S)

    if not completed_frames:
        raise RuntimeError("ERA5: all batches failed — no data collected")

    result = pd.concat(completed_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["lat", "lon", "timestamp"])
    return result


def build_thailand_grid(spacing_deg: float = 0.25) -> tuple[List[float], List[float]]:
    """Generate regular grid points covering Thailand bounding box."""
    lats = list(np.arange(5.5, 20.5, spacing_deg))
    lons = list(np.arange(97.5, 105.7, spacing_deg))
    # Build (lat, lon) pairs as a meshgrid: all lons for each lat
    lat_list = [lat for lat in lats for _ in lons]
    lon_list = [lon for _ in lats for lon in lons]
    return lat_list, lon_list
