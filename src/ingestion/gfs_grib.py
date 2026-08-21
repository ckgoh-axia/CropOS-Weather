"""GRIB addressing and leakage-safe run selection for NOAA GFS on AWS.

This module is pure logic — no network I/O — because the run-selection rule is
the correctness-critical part of the whole pipeline and must be testable
without a network.

Leakage rule (spec §4.3)
------------------------
Every forecast field must come from a model run whose PUBLICATION time is at
or before the issuance time. Initialisation time is not sufficient: GFS
publishes roughly 4 hours after initialisation, so a run initialised at 06Z is
not available to a forecast issued at 09Z.

Getting this wrong is silent. Validation scores rise and production fails,
because the run the model was trained to expect does not exist yet at
inference time.

Accumulation convention
-----------------------
GFS ``APCP`` accumulates from the last multiple of 6 hours, NOT from the run
start and NOT over a fixed window. Verified against real .idx files:

    f021 -> "18-21 hour acc fcst"   (3 h)
    f024 -> "18-24 hour acc fcst"   (6 h)
    f025 -> "24-25 hour acc fcst"   (1 h)

Mixing these silently compares a 6-hour total against a 1-hour total.
"""
from __future__ import annotations

import pandas as pd

# GFS runs 4x daily and publishes ~4 h after initialisation.
RUN_INTERVAL_H: int = 6
PUBLICATION_LAG_H: int = 4

# The bucket switched to a gfs.YYYYMMDD/HH/atmos/ layout on this date.
_ATMOS_LAYOUT_FROM = pd.Timestamp("2021-03-23", tz="UTC")


def select_run(
    issuance: pd.Timestamp,
    valid_time: pd.Timestamp,
    publication_lag_h: int = PUBLICATION_LAG_H,
) -> tuple[pd.Timestamp, int]:
    """Return the freshest run usable at ``issuance``, and its lead hours.

    Args:
        issuance:   The time the forecast is issued. Only runs published at or
                    before this instant may be used.
        valid_time: The time the forecast is valid for.
        publication_lag_h: Hours between run initialisation and availability.

    Returns:
        (run_init_time, lead_hours) where lead_hours = valid_time - run.

    Raises:
        ValueError: if valid_time is not after the selected run.
    """
    # Latest 6-hourly run whose publication time is <= issuance.
    latest_publishable = issuance - pd.Timedelta(hours=publication_lag_h)
    run = latest_publishable.floor(f"{RUN_INTERVAL_H}h")

    lead = int((valid_time - run).total_seconds() // 3600)
    if lead <= 0:
        raise ValueError(
            f"valid_time {valid_time} is not after selected run {run} "
            f"(issuance {issuance}, lag {publication_lag_h} h)"
        )
    return run, lead


def bucket_start(lead_h: int) -> int:
    """Return the lead hour at which this APCP accumulation bucket started."""
    if lead_h <= 0:
        raise ValueError(f"lead_h must be positive, got {lead_h}")
    return RUN_INTERVAL_H * ((lead_h - 1) // RUN_INTERVAL_H)


def gfs_key(run: pd.Timestamp, lead_h: int) -> str:
    """Return the S3 key for a GFS pgrb2 0.25-degree file."""
    day = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    name = f"gfs.t{hh}z.pgrb2.0p25.f{lead_h:03d}"
    if run >= _ATMOS_LAYOUT_FROM:
        return f"gfs.{day}/{hh}/atmos/{name}"
    return f"gfs.{day}/{hh}/{name}"


def graphcast_key(run: pd.Timestamp, lead_h: int) -> str:
    """Return the S3 key for an operational GraphCast-GFS pgrb2 file."""
    day = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    return (
        f"graphcastgfs.{day}/{hh}/forecasts_13_levels/"
        f"graphcastgfs.t{hh}z.pgrb2.0p25.f{lead_h:03d}"
    )
