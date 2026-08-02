# tests/unit/test_dataset.py
"""Unit tests for src/features/dataset.py

These tests use small synthetic DataFrames so no real parquet files or
HuggingFace credentials are required.  The ERA5 radius is set to a large value
(9999 km) so all synthetic grid points are always included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
<<<<<<< ours
import pytest
=======
>>>>>>> theirs
import torch
from torch_geometric.data import HeteroData

from src.features.dataset import CropOSDataset, _build_era5_label_df, _filter_era5_by_radius
from src.features.engineer import ERA5_SURFACE_VARS

<<<<<<< ours

=======
>>>>>>> theirs
# ── helpers ───────────────────────────────────────────────────────────────────

STATION_COORDS = {
    "STA0": (13.0, 100.0),
    "STA1": (14.0, 101.0),
}
STATION_ORDER = ["STA0", "STA1"]
HORIZONS_H = [12, 24]
N_HOURS = 50          # training timestamps (need ≥ max_horizon + a few for valid_ts)
NWP_VARS = [f"nwp_feat_{i}" for i in range(4)]   # 4 NWP features for speed


def _make_era5_df(n_hours: int = N_HOURS, n_pts: int = 6) -> pd.DataFrame:
    """Tiny ERA5 DataFrame: n_pts grid points × n_hours timestamps."""
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    lats = [13.0, 13.5, 14.0, 13.0, 13.5, 14.0][:n_pts]
    lons = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5][:n_pts]
    rng = np.random.default_rng(0)
    rows = []
    for ts in timestamps:
<<<<<<< ours
        for lat, lon in zip(lats, lons):
=======
        for lat, lon in zip(lats, lons, strict=True):
>>>>>>> theirs
            row: dict = {"timestamp": ts, "lat": lat, "lon": lon}
            for var in ERA5_SURFACE_VARS:
                row[var] = float(rng.uniform(0, 5))
            # Make precipitation non-trivial for label tests
            row["precipitation"] = float(rng.uniform(0, 5))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_nwp_df(n_hours: int = N_HOURS) -> pd.DataFrame:
    """Tiny NWP DataFrame: 2 stations × n_hours timestamps × 4 features."""
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    rows = []
    for ts in timestamps:
        for station, (lat, lon) in STATION_COORDS.items():
            row: dict = {"station": station, "timestamp": ts, "lat": lat, "lon": lon}
            for col in NWP_VARS:
                row[col] = float(rng.uniform(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_dataset(**kwargs) -> CropOSDataset:
    defaults = dict(
        era5_df=_make_era5_df(),
        nwp_df=_make_nwp_df(),
        station_order=STATION_ORDER,
        station_coords=STATION_COORDS,
        nwp_var_cols=NWP_VARS,
        horizons_h=HORIZONS_H,
        era5_node_radius_km=9999.0,   # include all synthetic points
        threshold_mm=1.0,
    )
    defaults.update(kwargs)
    return CropOSDataset(**defaults)


# ── filter helper ─────────────────────────────────────────────────────────────

def test_filter_era5_by_radius_includes_near_points():
    era5 = _make_era5_df(n_pts=6)
    # Only keep points within 60 km of STA0 (13.0, 100.0) — roughly the first 1-2 pts
    filtered = _filter_era5_by_radius(era5, {"STA0": (13.0, 100.0)}, radius_km=60.0)
    unique_pts = filtered[["lat", "lon"]].drop_duplicates()
    # 13.0/100.0 must survive; 14.0/102.5 should be filtered out (too far)
    assert ((unique_pts["lat"] == 13.0) & (unique_pts["lon"] == 100.0)).any()


def test_filter_era5_by_radius_large_radius_keeps_all():
    era5 = _make_era5_df(n_pts=6)
    filtered = _filter_era5_by_radius(era5, STATION_COORDS, radius_km=9999.0)
    before = era5[["lat", "lon"]].drop_duplicates().shape[0]
    after = filtered[["lat", "lon"]].drop_duplicates().shape[0]
    assert after == before


# ── label builder ─────────────────────────────────────────────────────────────

def test_build_era5_label_df_shape():
    era5 = _make_era5_df(n_hours=5)
    ldf = _build_era5_label_df(era5, STATION_COORDS, STATION_ORDER)
    # Each of 2 stations × 5 timestamps = 10 rows
    assert len(ldf) == 2 * 5
    assert set(ldf.columns) == {"timestamp", "station", "precip_mm"}


def test_build_era5_label_df_non_negative_precip():
    era5 = _make_era5_df(n_hours=10)
    ldf = _build_era5_label_df(era5, STATION_COORDS, STATION_ORDER)
    assert (ldf["precip_mm"] >= 0).all()


# ── CropOSDataset ─────────────────────────────────────────────────────────────

def test_dataset_len_positive():
    ds = _make_dataset()
    assert len(ds) > 0


def test_dataset_len_bounded_by_max_horizon():
    """We can only train on timestamps where t + max_horizon labels exist."""
    ds = _make_dataset()
    assert len(ds) <= N_HOURS - max(HORIZONS_H)


def test_dataset_getitem_returns_heterodata():
    ds = _make_dataset()
    sample = ds[0]
    assert isinstance(sample, HeteroData)


def test_dataset_era5_node_shape():
    ds = _make_dataset()
    sample = ds[0]
    n_era5 = ds.n_era5_nodes
    n_feat = ds.n_era5_features
    assert sample["era5"].x.shape == (n_era5, n_feat)


def test_dataset_era5_features_count():
    """era5 nodes must have 11 features (7 surface + 4 temporal)."""
    ds = _make_dataset()
    assert ds.n_era5_features == 11


def test_dataset_local_station_shape():
    ds = _make_dataset()
    sample = ds[0]
    assert sample["local_station"].x.shape == (len(STATION_ORDER), len(NWP_VARS))


def test_dataset_farm_placeholder_zeros():
    ds = _make_dataset()
    sample = ds[0]
    assert sample["farm"].x.shape == (len(STATION_ORDER), 1)
    assert (sample["farm"].x == 0).all()


def test_dataset_labels_shape():
    ds = _make_dataset()
    sample = ds[0]
    assert sample["farm"].y.shape == (len(STATION_ORDER), len(HORIZONS_H))


def test_dataset_labels_binary():
    """All label values must be 0 or 1."""
    ds = _make_dataset()
    for i in range(min(5, len(ds))):
        y = ds[i]["farm"].y
        assert torch.all((y == 0) | (y == 1)), f"Non-binary label at index {i}: {y}"


def test_dataset_edge_indices_present():
    ds = _make_dataset()
    sample = ds[0]
    assert hasattr(sample["era5", "to", "local_station"], "edge_index")
    assert hasattr(sample["era5", "to", "farm"], "edge_index")
    assert hasattr(sample["local_station", "to", "farm"], "edge_index")


def test_dataset_edge_index_dtype():
    ds = _make_dataset()
    sample = ds[0]
    ei = sample["era5", "to", "farm"].edge_index
    assert ei.dtype == torch.long


def test_dataset_era5_features_float32():
    ds = _make_dataset()
    sample = ds[0]
    assert sample["era5"].x.dtype == torch.float32


def test_dataset_nwp_no_nan():
    """NWP missing values are filled with 0 — no NaN should reach the tensor."""
    ds = _make_dataset()
    for i in range(min(3, len(ds))):
        assert not torch.any(torch.isnan(ds[i]["local_station"].x))


def test_dataset_different_horizons():
    """Dataset with [6, 12] horizons should produce labels of matching shape."""
    ds = _make_dataset(horizons_h=[6, 12])
    sample = ds[0]
    assert sample["farm"].y.shape[1] == 2


def test_dataset_high_threshold_all_no_rain():
    """With a very high threshold no timestamp should be labelled as rain."""
    ds = _make_dataset(threshold_mm=1e6)
    # ERA5 precipitation in synthetic data is 0–5 mm → all below 1e6
    for i in range(min(5, len(ds))):
        assert (ds[i]["farm"].y == 0).all()


def test_dataset_zero_threshold_mostly_rain():
    """With threshold=0 almost all labels should be rain (precip > 0)."""
    ds = _make_dataset(threshold_mm=0.0)
    rain_count = sum(int(ds[i]["farm"].y.sum()) for i in range(min(5, len(ds))))
    total = 5 * len(STATION_ORDER) * len(HORIZONS_H)
    assert rain_count > total * 0.5, "Expected mostly rain with threshold=0"
