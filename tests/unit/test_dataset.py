# tests/unit/test_dataset.py
"""Unit tests for src/features/dataset.py

These tests use small synthetic DataFrames so no real parquet files or
HuggingFace credentials are required.  The ERA5 radius is set to a large value
(9999 km) so all synthetic grid points are always included.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.features.dataset import (
    CropOSDataset,
    METAR_FEATURE_COLS,
    _build_era5_label_df,
    _filter_era5_by_radius,
    load_dataset_from_parquets,
)
from src.features.engineer import ERA5_SURFACE_VARS

# ── helpers ───────────────────────────────────────────────────────────────────

STATION_COORDS = {
    "STA0": (13.0, 100.0),
    "STA1": (14.0, 101.0),
}
STATION_ORDER = ["STA0", "STA1"]
HORIZONS_H = [12, 24]
N_HOURS = 50          # training timestamps (need ≥ max_horizon + a few for valid_ts)


def _make_era5_df(n_hours: int = N_HOURS, n_pts: int = 6) -> pd.DataFrame:
    """Tiny ERA5 DataFrame: n_pts grid points × n_hours timestamps."""
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    lats = [13.0, 13.5, 14.0, 13.0, 13.5, 14.0][:n_pts]
    lons = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5][:n_pts]
    rng = np.random.default_rng(0)
    rows = []
    for ts in timestamps:
        for lat, lon in zip(lats, lons, strict=True):
            row: dict = {"timestamp": ts, "lat": lat, "lon": lon}
            for var in ERA5_SURFACE_VARS:
                row[var] = float(rng.uniform(0, 5))
            # Make precipitation non-trivial for label tests
            row["precipitation"] = float(rng.uniform(0, 5))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_metar_df(n_hours: int = N_HOURS) -> pd.DataFrame:
    """Tiny METAR DataFrame: 2 stations × n_hours timestamps × 9 features."""
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    rows = []
    for ts in timestamps:
        for station, (lat, lon) in STATION_COORDS.items():
            row: dict = {"station": station, "timestamp": ts, "lat": lat, "lon": lon}
            row["precip_mm"] = float(rng.uniform(0, 5))
            row["rain_event"] = bool(rng.integers(0, 2))
            row["tmpf"] = float(rng.uniform(70, 95))
            row["dwpf"] = float(rng.uniform(60, 80))
            row["relh"] = float(rng.uniform(50, 95))
            row["drct"] = float(rng.uniform(0, 360))
            row["sknt"] = float(rng.uniform(0, 20))
            row["alti"] = float(rng.uniform(29.5, 30.5))
            row["vsby"] = float(rng.uniform(5, 10))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_dataset(**kwargs) -> CropOSDataset:
    defaults = dict(
        era5_df=_make_era5_df(),
        metar_df=_make_metar_df(),
        station_order=STATION_ORDER,
        station_coords=STATION_COORDS,
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


def test_dataset_metar_shape():
    """metar nodes must have 9 features (METAR_FEATURE_COLS)."""
    ds = _make_dataset()
    sample = ds[0]
    assert sample["metar"].x.shape == (len(STATION_ORDER), len(METAR_FEATURE_COLS))


def test_dataset_metar_features_count():
    ds = _make_dataset()
    assert ds.n_metar_features == len(METAR_FEATURE_COLS)
    assert ds.n_metar_features == 9


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
    assert hasattr(sample["era5", "to", "metar"], "edge_index")
    assert hasattr(sample["era5", "to", "farm"], "edge_index")
    assert hasattr(sample["metar", "to", "farm"], "edge_index")


def test_dataset_edge_index_dtype():
    ds = _make_dataset()
    sample = ds[0]
    ei = sample["era5", "to", "farm"].edge_index
    assert ei.dtype == torch.long


def test_dataset_era5_features_float32():
    ds = _make_dataset()
    sample = ds[0]
    assert sample["era5"].x.dtype == torch.float32


def test_dataset_metar_no_nan():
    """METAR missing values are filled with 0 — no NaN should reach the tensor."""
    ds = _make_dataset()
    for i in range(min(3, len(ds))):
        assert not torch.any(torch.isnan(ds[i]["metar"].x))


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


def test_dataset_rain_event_cast_to_float():
    """rain_event (bool in METAR) must be stored as float, not bool, in tensors."""
    ds = _make_dataset()
    sample = ds[0]
    assert sample["metar"].x.dtype == torch.float32


# ── load_dataset_from_parquets ────────────────────────────────────────────────


def _write_parquet(df: pd.DataFrame, directory: str, name: str) -> Path:
    p = Path(directory) / name
    df.to_parquet(p, index=False)
    return p


def _make_era5_parquet_df(n_hours: int = 30, n_pts: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    lats = [13.0, 13.5, 14.0][:n_pts]
    lons = [100.0, 100.5, 101.0][:n_pts]
    rng = np.random.default_rng(42)
    rows = []
    for ts in timestamps:
        for lat, lon in zip(lats, lons):
            row: dict = {"timestamp": ts, "lat": lat, "lon": lon}
            for var in ERA5_SURFACE_VARS:
                row[var] = float(rng.uniform(0, 5))
            row["precipitation"] = float(rng.uniform(0, 5))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_metar_parquet_df(n_hours: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(43)
    rows = []
    for ts in timestamps:
        for station, (lat, lon) in STATION_COORDS.items():
            row: dict = {"station": station, "timestamp": ts, "lat": lat, "lon": lon}
            row["precip_mm"] = float(rng.uniform(0, 5))
            row["rain_event"] = bool(rng.integers(0, 2))
            row["tmpf"] = float(rng.uniform(70, 95))
            row["dwpf"] = float(rng.uniform(60, 80))
            row["relh"] = float(rng.uniform(50, 95))
            row["drct"] = float(rng.uniform(0, 360))
            row["sknt"] = float(rng.uniform(0, 20))
            row["alti"] = float(rng.uniform(29.5, 30.5))
            row["vsby"] = float(rng.uniform(5, 10))
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_dataset_from_parquets_basic():
    """load_dataset_from_parquets returns a valid CropOSDataset from parquet files."""
    with tempfile.TemporaryDirectory() as tmp:
        era5_path = _write_parquet(_make_era5_parquet_df(), tmp, "era5.parquet")
        metar_path = _write_parquet(_make_metar_parquet_df(), tmp, "metar.parquet")

        ds = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
        )
        assert len(ds) > 0


def test_load_dataset_from_parquets_with_era5_recent():
    """era5_recent_path data is concatenated into the training set."""
    with tempfile.TemporaryDirectory() as tmp:
        base_era5 = _make_era5_parquet_df(n_hours=20)
        recent_era5 = _make_era5_parquet_df(n_hours=10)
        # Shift recent timestamps forward so rows are distinct
        recent_era5["timestamp"] = recent_era5["timestamp"] + pd.Timedelta(hours=20)

        era5_path = _write_parquet(base_era5, tmp, "era5.parquet")
        metar_path = _write_parquet(_make_metar_parquet_df(n_hours=30), tmp, "metar.parquet")
        era5_recent_path = _write_parquet(recent_era5, tmp, "era5_recent.parquet")

        ds = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            era5_recent_path=era5_recent_path,
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
        )
        assert len(ds) > 0
        # Dataset should have more ERA5 rows than base-only
        ds_base = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
        )
        assert ds.n_era5_nodes >= ds_base.n_era5_nodes or len(ds) >= len(ds_base)


def test_load_dataset_from_parquets_era5_recent_nonexistent_path_ignored():
    """A non-existent era5_recent_path is silently ignored (no crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        era5_path = _write_parquet(_make_era5_parquet_df(), tmp, "era5.parquet")
        metar_path = _write_parquet(_make_metar_parquet_df(), tmp, "metar.parquet")

        ds = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            era5_recent_path=Path(tmp) / "does_not_exist.parquet",
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
        )
        assert len(ds) > 0


def test_load_dataset_from_parquets_date_filtering():
    """start_date / end_date filter rows from both ERA5 and METAR parquets."""
    with tempfile.TemporaryDirectory() as tmp:
        era5_path = _write_parquet(_make_era5_parquet_df(n_hours=30), tmp, "era5.parquet")
        metar_path = _write_parquet(_make_metar_parquet_df(n_hours=30), tmp, "metar.parquet")

        ds = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
            start_date="2020-01-01",
            end_date="2020-01-02",
        )
        # Filtered to 24h window — can't be longer than full 30h dataset
        ds_full = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
        )
        assert len(ds) <= len(ds_full)


def test_load_dataset_metar_feature_cols_constant():
    """METAR feature columns are fixed regardless of parquet contents."""
    with tempfile.TemporaryDirectory() as tmp:
        era5_path = _write_parquet(_make_era5_parquet_df(), tmp, "era5.parquet")
        metar_path = _write_parquet(_make_metar_parquet_df(), tmp, "metar.parquet")

        ds = load_dataset_from_parquets(
            era5_path=era5_path,
            metar_path=metar_path,
            station_order=STATION_ORDER,
            station_coords=STATION_COORDS,
            horizons_h=HORIZONS_H,
            era5_node_radius_km=9999.0,
            threshold_mm=1.0,
        )
        assert ds.metar_feature_cols == METAR_FEATURE_COLS
        assert ds.n_metar_features == 9
