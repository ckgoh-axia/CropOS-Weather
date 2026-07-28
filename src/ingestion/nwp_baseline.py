"""GFS NWP feature ingestion via Open-Meteo historical forecast API.

Production architecture: CropOS is a GFS-correction model.
  - Training input  : historical GFS forecasts (this file)
  - Training labels : ERA5 / METAR precipitation (ground truth)
  - Inference input : live GFS forecast from Open-Meteo forecast API
                      (same variables, same naming, zero train/serve mismatch)

The historical forecast API is distinct from the archive API:
  - URL   : historical-forecast-api.open-meteo.com  (not archive-api)
  - Model : gfs_seamless (GFS + GEFS blend, 0.25° grid)
  - Limit : available from 2016-01-01 onward
  - Rate  : more lenient than ERA5 archive; 5s between stations is sufficient

Checkpoint design: one parquet file per station stored in a checkpoint dir.
If a station file exists, that station is skipped on re-run. This means
partial failures (network timeout, rate limit) do not force a full restart.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)

NWP_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
NWP_MODEL = "gfs_seamless"

# Delay between station requests — the historical forecast API is more lenient
# than the ERA5 archive API, but 10s is polite and keeps us well under any limit.
STATION_DELAY_S = 10.0

# Default full variable set — mirrors configs/data.yaml nwp_variables.
# This list is the single source of truth when called without a config.
NWP_DEFAULT_VARIABLES: List[str] = [
    # Surface — identical naming to ERA5 for clean train/serve feature parity
    "temperature_2m",
    "dewpoint_2m",
    "relativehumidity_2m",
    "precipitation",
    "windspeed_10m",
    "winddirection_10m",
    "surface_pressure",
    # Convective instability — primary drivers of tropical rainfall
    "cape",
    "lifted_index",
    # Moisture column — full depth, not just surface dewpoint
    "precipitable_water",
    # Cloud layer structure — separate low/mid/high rather than total cover.
    # Low = boundary layer, Mid = congestus/altostratus, High = cirrus/anvil.
    # This separates "humid but capped" from "organised deep convection".
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    # Diurnal heating — tropical convection is timing-driven.
    # Actual insolation, not just hour-of-day proxy.
    "shortwave_radiation",
    # 700 hPa — mid-level steering; distinguishes slow from fast-moving systems
    "windspeed_700hPa",
    "winddirection_700hPa",
    # 850 hPa — low-level jet, monsoon moisture transport, BL convergence
    "windspeed_850hPa",
    "winddirection_850hPa",
    "temperature_850hPa",
    # 500 hPa — mid-troposphere blocking, trough/ridge position
    "geopotential_height_500hPa",
    # 200 hPa — jet stream, upper divergence, storm outflow ventilation
    "windspeed_200hPa",
    "winddirection_200hPa",
    # Land surface — soil moisture feeds afternoon convection
    "soil_moisture_0_to_7cm",
]


def _build_client() -> openmeteo_requests.Client:
    session = requests_cache.CachedSession(".cache/nwp", expire_after=-1)
    session = retry(session, retries=5, backoff_factor=2.0)  # 2s, 4s, 8s, 16s, 32s
    return openmeteo_requests.Client(session=session)


def fetch_nwp_at_station(
    station: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: List[str] | None = None,
) -> pd.DataFrame:
    """Fetch full NWP feature set for one station across the full date range.

    Args:
        station  : ICAO station ID (used only for logging and 'station' column)
        lat, lon : station coordinates
        start_date, end_date : ISO date strings (YYYY-MM-DD)
        variables: list of Open-Meteo variable names; defaults to NWP_DEFAULT_VARIABLES

    Returns:
        DataFrame with columns: station, timestamp, lat, lon, <all nwp variables>
        Variables that the API could not return are absent (logged at WARNING).
    """
    if variables is None:
        variables = NWP_DEFAULT_VARIABLES

    client = _build_client()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        "models": NWP_MODEL,
        "timezone": "UTC",
    }

    try:
        responses = client.weather_api(NWP_URL, params=params)
    except Exception as exc:
        logger.error(f"NWP | {station}: API call failed — {exc}")
        raise

    r = responses[0]
    hourly = r.Hourly()
    timestamps = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data: dict = {
        "station": station,
        "timestamp": timestamps,
        "lat": lat,
        "lon": lon,
    }

    n_returned = hourly.VariablesLength()
    if n_returned != len(variables):
        logger.warning(
            f"NWP | {station}: requested {len(variables)} variables, "
            f"API returned {n_returned}. "
            f"Some variables may be unavailable for {NWP_MODEL}."
        )

    for i in range(n_returned):
        var_name = variables[i] if i < len(variables) else f"unknown_{i}"
        data[f"nwp_{var_name}"] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)

    # Coerce all nwp_* columns to float to avoid object dtype / parquet issues
    nwp_cols = [c for c in df.columns if c.startswith("nwp_")]
    for col in nwp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        f"NWP | {station}: {len(df):,} rows, "
        f"{len(nwp_cols)} feature columns"
    )
    return df


def fetch_all_stations(
    station_coords: dict[str, tuple[float, float]],
    start_date: str,
    end_date: str,
    variables: List[str] | None = None,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch NWP features for all stations with per-station checkpointing.

    Args:
        station_coords : {station_id: (lat, lon)}
        start_date     : effective start (caller should clamp to nwp_min_start)
        end_date       : ISO date string
        variables      : feature list; defaults to NWP_DEFAULT_VARIABLES
        checkpoint_dir : directory for per-station parquet checkpoints;
                         None disables checkpointing.

    Returns:
        Concatenated DataFrame of all stations.
    """
    if variables is None:
        variables = NWP_DEFAULT_VARIABLES

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    stations = list(station_coords.items())

    for idx, (station, (lat, lon)) in enumerate(stations):
        # Check checkpoint — skip station if already downloaded.
        if checkpoint_dir is not None:
            ckpt_file = checkpoint_dir / f"{station}.parquet"
            if ckpt_file.exists():
                df = pd.read_parquet(ckpt_file)
                frames.append(df)
                logger.info(f"NWP | {station} ({idx+1}/{len(stations)}): restored from checkpoint ({len(df):,} rows)")
                continue

        logger.info(f"NWP | {station} ({idx+1}/{len(stations)}): fetching...")
        try:
            df = fetch_nwp_at_station(station, lat, lon, start_date, end_date, variables)
            frames.append(df)
            # Persist checkpoint immediately so a mid-run failure doesn't lose this station.
            if checkpoint_dir is not None:
                df.to_parquet(checkpoint_dir / f"{station}.parquet", index=False)
                logger.info(f"NWP | {station}: checkpoint saved")
        except Exception as exc:
            logger.error(f"NWP | {station}: FAILED — {exc} (will retry on next run)")

        if idx < len(stations) - 1:
            time.sleep(STATION_DELAY_S)

    if not frames:
        raise RuntimeError("NWP: all stations failed — no data collected")

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["station", "timestamp"]).reset_index(drop=True)
    logger.info(
        f"NWP | total: {len(result):,} rows across "
        f"{result['station'].nunique()} stations"
    )
    return result


# ---------------------------------------------------------------------------
# Legacy single-point entry point (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fetch_nwp_at_point(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Legacy single-point fetch — returns only precipitation for backward compat."""
    df = fetch_nwp_at_station("_point", lat, lon, start_date, end_date, ["precipitation"])
    return df.rename(columns={"nwp_precipitation": "nwp_precip_mm"})
