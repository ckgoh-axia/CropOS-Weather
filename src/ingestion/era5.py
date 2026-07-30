"""ERA5-Land ingestion via Open-Meteo archive API.

Checkpoint design: one parquet file per batch stored in checkpoint_dir.
  checkpoint_dir/batch_0000.parquet   ← 10 grid points × full date range
  checkpoint_dir/batch_0001.parquet
  ...

This avoids the memory spike that occurs when a single growing checkpoint file
is read and rewritten after each batch.  With the old approach, by batch ~52
the in-memory concat reached ~6 GB (36 M rows × 2 due to temporary allocation
during pd.concat), which killed the GitHub Actions runner via OOM before the
HF checkpoint push could run — leaving HF stuck and every re-run repeating the
same batches.

With per-batch files:
  - Peak RAM per batch ≈ one_batch_size (~28 MB) instead of O(all_rows).
  - HF push targets only the new batch files written this run.
  - max_new_batches stops the loop cleanly before the rate-limit window closes.
  - Era5PartialDownload is raised (not RuntimeError) when the run ends before
    all batches are done — callers skip the final assembly to avoid OOM.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)


class Era5PartialDownload(RuntimeError):
    """Raised by fetch_era5_grid when the run ends before all batches are done.

    This is not an error — checkpoints for completed batches have been saved.
    Callers should skip the expensive final-assembly step and exit cleanly so
    the next cron run can resume from the checkpoint without OOM risk.
    """


ERA5_VARIABLES = [
    "temperature_2m", "dewpoint_2m", "relativehumidity_2m",
    "precipitation", "windspeed_10m", "winddirection_10m",
    "surface_pressure",
]
ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"

# archive-api.open-meteo.com free tier: ~2 requests/minute.
# 65 s between batches keeps us comfortably under that limit.
# 198 batches × ~70 s each ≈ 3.9 h — within GitHub Actions 6-hour cap.
BATCH_SIZE = 10
BATCH_DELAY_S = 65.0


def _build_client() -> openmeteo_requests.Client:
    session = requests_cache.CachedSession(".cache/era5", expire_after=-1)
    session = retry(session, retries=5, backoff_factor=2.0)  # 2 s, 4 s, 8 s, 16 s, 32 s
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
    for r, lat, lon in zip(responses, lats, lons, strict=False):
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
    max_new_batches: int | None = None,
) -> pd.DataFrame:
    """Fetch ERA5-Land for all grid points using batched API calls.

    Checkpoint files — one parquet per batch — are written immediately after
    each successful batch so a re-run resumes exactly where it stopped without
    loading any previously-downloaded data into memory during the download loop.

    Args:
        checkpoint_dir:  Directory for per-batch parquet files.
                         Pass None to disable checkpointing (test-only path).
        max_new_batches: Stop after downloading this many new batches and
                         return, even if more remain.  Used by cron runs to
                         guarantee a clean shutdown before the rate-limit window
                         closes.  None = download until done or until a batch
                         fails.
    """
    client = _build_client()
    pairs = list(zip(lat_points, lon_points, strict=False))
    n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    use_checkpoint = checkpoint_dir is not None

    if use_checkpoint:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Determine which batches are already done.
    # With per-batch files: just check file existence — no data loaded here.
    # -----------------------------------------------------------------------
    completed_batches: set[int] = set()
    if use_checkpoint:
        for idx in range(n_batches):
            if (checkpoint_dir / f"batch_{idx:04d}.parquet").exists():
                completed_batches.add(idx)
        if completed_batches:
            logger.info(
                f"ERA5 | resuming: {len(completed_batches)}/{n_batches} batches already done"
            )

    # -----------------------------------------------------------------------
    # Download loop
    # -----------------------------------------------------------------------
    new_batches_downloaded = 0
    in_memory_frames: List[pd.DataFrame] = []  # only used when use_checkpoint=False

    for i in range(0, len(pairs), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE
        batch_num = batch_idx + 1

        if batch_idx in completed_batches:
            logger.info(f"ERA5 | batch {batch_num}/{n_batches} — skipped (checkpoint)")
            continue

        if max_new_batches is not None and new_batches_downloaded >= max_new_batches:
            logger.info(
                f"ERA5 | reached max_new_batches={max_new_batches} — "
                f"stopping cleanly to allow checkpoint push"
            )
            break

        batch = pairs[i: i + BATCH_SIZE]
        b_lats = [p[0] for p in batch]
        b_lons = [p[1] for p in batch]
        logger.info(
            f"ERA5 | batch {batch_num}/{n_batches} ({len(batch)} points, "
            f"lat {min(b_lats):.2f}–{max(b_lats):.2f})"
        )

        try:
            frame = _fetch_batch(client, b_lats, b_lons, start_date, end_date)
            if use_checkpoint:
                # Write only this batch — peak RAM = one_batch, not all_batches.
                frame.to_parquet(
                    checkpoint_dir / f"batch_{batch_idx:04d}.parquet", index=False
                )
            else:
                in_memory_frames.append(frame)
            new_batches_downloaded += 1
            logger.info(f"ERA5 | batch {batch_num}/{n_batches} done ({len(frame):,} rows)")
        except Exception as exc:
            logger.error(f"ERA5 | batch {batch_num}/{n_batches} FAILED: {exc}")
            logger.error("ERA5 | stopping — re-run the script to resume from checkpoint")
            break

        if i + BATCH_SIZE < len(pairs):
            time.sleep(BATCH_DELAY_S)

    # -----------------------------------------------------------------------
    # Assemble final result — only when ALL batches are complete.
    # -----------------------------------------------------------------------
    if use_checkpoint:
        batch_files = sorted(checkpoint_dir.glob("batch_*.parquet"))
        if not batch_files:
            raise RuntimeError("ERA5: all batches failed — no data collected")
        done_count = len(batch_files)
        if done_count < n_batches:
            # Not all batches are done.  Raise Era5PartialDownload instead of
            # attempting a partial concat — reading 80-190 batch files into memory
            # in a single pd.concat can exceed the 7 GB GitHub Actions RAM limit.
            # The caller's finally block will push the new checkpoint files to HF
            # and then exit cleanly (exit code 0, green run).
            raise Era5PartialDownload(
                f"ERA5: {done_count}/{n_batches} batches done — "
                f"checkpoint saved, re-run to continue"
            )
        # All batches complete — assemble the full dataset once.
        # Peak RAM here = total checkpoint size (~28 MB × n_batches ≈ 5.5 GB),
        # but this only runs on the final run, not on every intermediate cron run.
        logger.info(f"ERA5 | all {n_batches} batches complete — assembling final dataset")
        result = pd.concat(
            [pd.read_parquet(f) for f in batch_files],
            ignore_index=True,
        )
    else:
        # No-checkpoint path (unit tests only — small mock data, no OOM risk).
        if not in_memory_frames:
            raise RuntimeError("ERA5: all batches failed — no data collected")
        result = pd.concat(in_memory_frames, ignore_index=True)

    result = result.drop_duplicates(subset=["lat", "lon", "timestamp"])
    return result


def build_thailand_grid(spacing_deg: float = 0.25) -> tuple[List[float], List[float]]:
    """Generate regular grid points covering Thailand bounding box."""
    lats = list(np.arange(5.5, 20.5, spacing_deg))
    lons = list(np.arange(97.5, 105.7, spacing_deg))
    lat_list = [lat for lat in lats for _ in lons]
    lon_list = [lon for _ in lats for lon in lons]
    return lat_list, lon_list
