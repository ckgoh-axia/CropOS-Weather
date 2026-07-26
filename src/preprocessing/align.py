"""Align all data sources to hourly UTC."""
from __future__ import annotations

import pandas as pd


def align_to_hourly_utc(
    df: pd.DataFrame,
    value_col: str,
    agg: str = "sum",
) -> pd.DataFrame:
    """Resample a DataFrame to hourly UTC, aggregating value_col."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    resampled = df[[value_col]].resample("h").agg(agg)
    return resampled.reset_index()


def merge_era5_metar(
    era5_df: pd.DataFrame,
    metar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-join ERA5 grid and METAR station data on timestamp (outer join)."""
    era5_df = era5_df.copy()
    metar_df = metar_df.copy()
    era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"], utc=True)
    metar_df["timestamp"] = pd.to_datetime(metar_df["timestamp"], utc=True)
    # Merge on timestamp only — graph builder handles spatial proximity
    merged = pd.merge(era5_df, metar_df, on="timestamp", how="outer", suffixes=("_era5", "_metar"))
    return merged.reset_index(drop=True)
