"""GPM IMERG ingestion via gpm-api."""
from __future__ import annotations
import xarray as xr
import numpy as np
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)


def build_thailand_bbox() -> Dict[str, float]:
    return {"lat_min": 5.5, "lat_max": 20.5, "lon_min": 97.5, "lon_max": 105.7}


def download_imerg(start_date: str, end_date: str, data_dir: str = "data/raw/imerg") -> None:
    """Download GPM IMERG Final Run for Thailand bounding box."""
    import gpm
    bbox = build_thailand_bbox()
    os.makedirs(data_dir, exist_ok=True)
    gpm.download(
        product="IMERG-FR",
        product_type="RS",
        start_time=start_date,
        end_time=end_date,
        bbox=[bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]],
        force_download=False,
    )
    logger.info(f"IMERG download complete: {start_date}→{end_date}")


def load_imerg(start_date: str, end_date: str) -> xr.Dataset:
    """Load downloaded IMERG files as xarray Dataset."""
    import gpm
    bbox = build_thailand_bbox()
    return gpm.open_dataset(
        product="IMERG-FR",
        product_type="RS",
        start_time=start_date,
        end_time=end_date,
        bbox=[bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]],
        variables=["precipitation"],
    )


def extract_imerg_at_points(
    ds: xr.Dataset,
    lat_points: List[float],
    lon_points: List[float],
) -> np.ndarray:
    """Extract IMERG precipitation at (lat, lon) points via nearest-neighbor. Returns (time, n_points)."""
    results = [
        ds["precipitation"].sel(lat=lat, lon=lon, method="nearest").values
        for lat, lon in zip(lat_points, lon_points)
    ]
    return np.stack(results, axis=-1)
