import numpy as np
import xarray as xr
from src.ingestion.imerg import build_thailand_bbox, extract_imerg_at_points

def test_build_thailand_bbox_values():
    bbox = build_thailand_bbox()
    assert bbox["lat_min"] == 5.5
    assert bbox["lat_max"] == 20.5
    assert bbox["lon_min"] == 97.5
    assert bbox["lon_max"] == 105.7

def test_extract_imerg_at_points_shape():
    fake_ds = xr.Dataset(
        {"precipitation": (["time", "lat", "lon"], np.random.rand(4, 10, 10))},
        coords={
            "time": np.arange(4),
            "lat": np.linspace(5.5, 20.5, 10),
            "lon": np.linspace(97.5, 105.7, 10),
        },
    )
    result = extract_imerg_at_points(fake_ds, [15.0, 16.5], [102.0, 103.0])
    assert result.shape == (4, 2)  # (time, n_points)
