"""Build precipitation labels for CropOS training.

Label priority:
  1. METAR p01i (hourly rain gauge, mm) at the 16 Thai airport stations — real
     observed precipitation, the ground truth we want the model to learn from.
  2. ERA5 precipitation (mm/h) everywhere else on the grid — a spatial fill so
     every farm target node has a label even without a co-located gauge.

This module merges the two sources into a single DataFrame keyed by
(lat, lon, timestamp) that the DataLoader can join to farm target nodes.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ERA5 precipitation is in m/h; convert to mm/h
ERA5_PRECIP_M_TO_MM = 1000.0


def build_labels(
    metar_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    precip_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    """Merge METAR gauge readings with ERA5 spatial fill into a label table.

    Args:
        metar_df: Output of fetch_all_thai_stations(). Must have columns:
                  station, timestamp, precip_mm (already in mm), lat, lon.
                  parse_metar_response() handles the p01i→mm conversion upstream.
        era5_df:  Output of fetch_era5_grid(). Must have columns:
                  lat, lon, timestamp, precipitation_sum (m/h or mm/h).
        precip_threshold_mm: Rain/no-rain threshold in mm. Default 1.0 mm/h.

    Returns:
        DataFrame with columns: lat, lon, timestamp, precip_mm, label_source,
        rain (bool), where label_source is 'metar' or 'era5'.
    """
    metar_labels = _extract_metar_labels(metar_df)
    era5_labels = _extract_era5_labels(era5_df)

    # METAR labels take priority: drop ERA5 rows at station locations/times
    metar_keys = set(zip(metar_labels["lat"].round(4), metar_labels["lon"].round(4),
                         metar_labels["timestamp"].astype(str), strict=False))
    era5_mask = era5_labels.apply(
        lambda r: (round(r["lat"], 4), round(r["lon"], 4), str(r["timestamp"])) not in metar_keys,
        axis=1,
    )
    merged = pd.concat([metar_labels, era5_labels[era5_mask]], ignore_index=True)
    merged["rain"] = merged["precip_mm"] >= precip_threshold_mm

    n_metar = (merged["label_source"] == "metar").sum()
    n_era5 = (merged["label_source"] == "era5").sum()
    logger.info(f"Labels: {n_metar:,} METAR rows + {n_era5:,} ERA5 fill rows "
                f"(threshold={precip_threshold_mm} mm, rain_frac="
                f"{merged['rain'].mean():.3f})")
    return merged


def _extract_metar_labels(metar_df: pd.DataFrame) -> pd.DataFrame:
    """Extract METAR precipitation labels from fetch_all_thai_stations() output.

    parse_metar_response() already converts p01i (inches) → precip_mm and
    renames 'valid' → 'timestamp', so we consume those columns directly.
    """
    df = metar_df.copy()

    # precip_mm is already in mm (converted from inches by parse_metar_response)
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop rows without a gauge reading
    df = df.dropna(subset=["precip_mm", "lat", "lon"])
    df["label_source"] = "metar"

    return df[["lat", "lon", "timestamp", "precip_mm", "label_source"]].reset_index(drop=True)


def _extract_era5_labels(era5_df: pd.DataFrame) -> pd.DataFrame:
    """Extract ERA5 precipitation and convert units to mm/h."""
    df = era5_df.copy()

    # precipitation_sum from Open-Meteo ERA5-Land is in mm (already, hourly)
    # but double-check: if values are tiny (<0.01 typical for m/h), convert
    if "precipitation_sum" in df.columns:
        precip_col = "precipitation_sum"
    elif "precipitation" in df.columns:
        precip_col = "precipitation"
    else:
        raise ValueError(
            f"No precipitation column found in ERA5 df. Columns: {df.columns.tolist()}"
        )

    df["precip_mm"] = pd.to_numeric(df[precip_col], errors="coerce").fillna(0.0)

    # ERA5 via Open-Meteo returns mm already, but guard against m/h values
    if df["precip_mm"].max() < 0.05 and df["precip_mm"].sum() > 0:
        logger.warning("ERA5 precip looks like m/h — converting × 1000")
        df["precip_mm"] = df["precip_mm"] * ERA5_PRECIP_M_TO_MM

    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["label_source"] = "era5"

    return df[["lat", "lon", "timestamp", "precip_mm", "label_source"]].reset_index(drop=True)
