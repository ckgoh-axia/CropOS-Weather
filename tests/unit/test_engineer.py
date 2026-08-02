# tests/unit/test_engineer.py
"""Unit tests for src/features/engineer.py"""
<<<<<<< ours
import math
=======
>>>>>>> theirs
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    DERIVED_NWP_FEATURES,
    ERA5_FEATURE_NAMES,
    ERA5_SURFACE_VARS,
    LOCAL_STATION_IN,
    REGION_COLS,
    REGIONS,
    STATIC_CONTINUOUS,
    STATIC_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    TERRAIN_CLASSES,
    TERRAIN_COLS,
    FeatureScaler,
    add_derived_nwp_features,
    add_station_static_features,
    add_temporal_features,
    prepare_era5_features,
    prepare_nwp_features,
)

<<<<<<< ours

=======
>>>>>>> theirs
# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_era5_df(n_rows: int = 24) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=n_rows, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    data: dict = {"timestamp": timestamps, "lat": 13.0, "lon": 100.0}
    for var in ERA5_SURFACE_VARS:
        data[var] = rng.uniform(0, 1, size=n_rows).astype(np.float32)
    return pd.DataFrame(data)


def _make_nwp_df(n_stations: int = 3, n_hours: int = 12) -> pd.DataFrame:
    """Minimal NWP df with synthetic nwp_var_* columns (no real column names)."""
    stations = [f"STA{i}" for i in range(n_stations)]
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    rows = []
    rng = np.random.default_rng(1)
    for ts in timestamps:
        for s in stations:
            row: dict = {"station": s, "timestamp": ts, "lat": 13.0, "lon": 100.0}
            for i in range(5):
                row[f"nwp_var_{i}"] = rng.uniform(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_full_nwp_df(n_stations: int = 2, n_hours: int = 30) -> pd.DataFrame:
    """NWP df with real column names needed for derived feature computation."""
    stations = [f"VTB{i}" for i in range(n_stations)]
    timestamps = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    rows = []
    rng = np.random.default_rng(42)
    for ts in timestamps:
        for s in stations:
            row: dict = {"station": s, "timestamp": ts, "lat": 13.0, "lon": 100.0}
            row["nwp_temperature_2m"]   = float(rng.uniform(25, 35))
            row["nwp_dewpoint_2m"]      = float(rng.uniform(18, 28))
            row["nwp_surface_pressure"] = float(rng.uniform(1000, 1020))
            row["nwp_precipitation"]    = float(rng.uniform(0, 5))
            row["nwp_windspeed_10m"]    = float(rng.uniform(0, 10))
            row["nwp_windspeed_850hPa"] = float(rng.uniform(5, 20))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_station_metadata(stations: list[str]) -> dict:
    """Build a minimal station_metadata dict for the given station ids."""
    terrain_choices = TERRAIN_CLASSES  # sorted: coastal, mountain, plain, urban, valley
    region_choices  = REGIONS          # sorted: central, north, northeast, south
    return {
        s: {
            "elevation_m":   float(50 * (i + 1)),
            "coast_km":      float(10 * (i + 1)),
            "terrain_class": terrain_choices[i % len(terrain_choices)],
            "region":        region_choices[i % len(region_choices)],
        }
        for i, s in enumerate(stations)
    }


# ── add_temporal_features ─────────────────────────────────────────────────────

def test_temporal_features_column_names():
    df = _make_era5_df(1)
    out = add_temporal_features(df)
    for col in TEMPORAL_FEATURE_NAMES:
        assert col in out.columns, f"Missing column: {col}"


def test_temporal_features_midnight_hour_zero():
    """At midnight UTC sin_hour=0, cos_hour=1."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2020-06-01 00:00", tz="UTC")]})
    out = add_temporal_features(df)
    assert abs(out["sin_hour"].iloc[0]) < 1e-5
    assert abs(out["cos_hour"].iloc[0] - 1.0) < 1e-5


def test_temporal_features_noon():
    """At noon sin_hour=0, cos_hour=-1."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2020-06-01 12:00", tz="UTC")]})
    out = add_temporal_features(df)
    assert abs(out["sin_hour"].iloc[0]) < 1e-5
    assert abs(out["cos_hour"].iloc[0] + 1.0) < 1e-5


def test_temporal_features_values_in_range():
    df = _make_era5_df(100)
    out = add_temporal_features(df)
    for col in TEMPORAL_FEATURE_NAMES:
        assert out[col].between(-1.0, 1.0).all(), f"{col} outside [-1, 1]"


def test_temporal_features_does_not_mutate():
    df = _make_era5_df(5)
    original_cols = set(df.columns)
    add_temporal_features(df)
    assert set(df.columns) == original_cols  # original unchanged


# ── prepare_era5_features ─────────────────────────────────────────────────────

def test_prepare_era5_features_output_columns():
    df = _make_era5_df(24)
    out = prepare_era5_features(df)
    expected = set(ERA5_FEATURE_NAMES + ["lat", "lon", "timestamp"])
    assert expected.issubset(set(out.columns))


def test_prepare_era5_features_count():
    """7 surface + 4 temporal = 11 total feature columns."""
    df = _make_era5_df(24)
    out = prepare_era5_features(df)
    feature_cols = [c for c in ERA5_FEATURE_NAMES if c in out.columns]
    assert len(feature_cols) == 11


def test_prepare_era5_features_missing_var_raises():
    df = _make_era5_df(5).drop(columns=["temperature_2m"])
    with pytest.raises(ValueError, match="temperature_2m"):
        prepare_era5_features(df)


def test_prepare_era5_features_preserves_row_count():
    df = _make_era5_df(48)
    assert len(prepare_era5_features(df)) == 48


# ── prepare_nwp_features ──────────────────────────────────────────────────────

def test_prepare_nwp_features_keeps_nwp_columns():
    df = _make_nwp_df()
    # add_derived=False so derived cols are not appended; add_static=False for same reason
    out = prepare_nwp_features(df, add_derived=False, add_static=False)
    nwp_cols = [c for c in out.columns if c.startswith("nwp_")]
    assert len(nwp_cols) == 5


def test_prepare_nwp_features_subset_by_name():
    df = _make_nwp_df()
    out = prepare_nwp_features(df, nwp_variables=["var_0", "var_1"],
                                add_derived=False, add_static=False)
    nwp_cols = [c for c in out.columns if c.startswith("nwp_")]
    assert sorted(nwp_cols) == ["nwp_var_0", "nwp_var_1"]


def test_prepare_nwp_features_missing_column_raises():
    df = _make_nwp_df()
    with pytest.raises(ValueError, match="nwp_nonexistent"):
        prepare_nwp_features(df, nwp_variables=["nonexistent"],
                             add_derived=False, add_static=False)


# ── add_derived_nwp_features ──────────────────────────────────────────────────

def test_derived_features_column_names():
    df = _make_full_nwp_df()
    out = add_derived_nwp_features(df)
    for col in DERIVED_NWP_FEATURES:
        assert col in out.columns, f"Missing derived column: {col}"


def test_derived_features_count():
    """Exactly 6 derived columns added (not more, not fewer)."""
    assert len(DERIVED_NWP_FEATURES) == 6


def test_derived_features_dewpoint_depression_formula():
    """nwp_dd = T2m − Td2m — verify the formula, not a physical constraint."""
    df = _make_full_nwp_df()
    out = add_derived_nwp_features(df)
    expected = (df["nwp_temperature_2m"] - df["nwp_dewpoint_2m"]).astype(np.float32)
    # Reindex expected to match output sort order (station, timestamp)
    out_sorted = out.reset_index(drop=True)
    expected_aligned = expected.reindex(
        df.sort_values(["station", "timestamp"]).index
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        out_sorted["nwp_dd"].values, expected_aligned.values, rtol=1e-5,
        err_msg="nwp_dd should equal nwp_temperature_2m - nwp_dewpoint_2m"
    )


def test_derived_features_no_cross_station_bleed():
    """Pressure tendency should not bleed across station boundaries.

    Station A row 0 → diff(3) should be NaN (filled to 0), not station B's data.
    """
    df = _make_full_nwp_df(n_stations=2, n_hours=6)
    out = add_derived_nwp_features(df, fill_value=0.0)
    # First 3 rows for each station are NaN → filled to 0
    for station in out["station"].unique():
        first_three = out[out["station"] == station].head(3)
        assert (first_three["nwp_dp_3h"] == 0.0).all(), (
            f"Station {station}: first dp_3h values should be 0 (filled NaN), "
            f"not {first_three['nwp_dp_3h'].values}"
        )


def test_derived_features_dp_3h_vs_dp_6h_independence():
    """dp_3h and dp_6h are computed independently and should differ."""
    df = _make_full_nwp_df(n_stations=1, n_hours=20)
    out = add_derived_nwp_features(df)
    # After the first 6 rows (which are NaN-filled), the two series should not be equal
    tail = out.iloc[6:]
    assert not (tail["nwp_dp_3h"] == tail["nwp_dp_6h"]).all()


def test_derived_features_precip_24h_sum_non_negative():
    df = _make_full_nwp_df()
    out = add_derived_nwp_features(df)
    assert (out["nwp_precip_24h_sum"] >= 0.0).all()


def test_derived_features_does_not_mutate_input():
    df = _make_full_nwp_df()
    original_cols = set(df.columns)
    add_derived_nwp_features(df)
    assert set(df.columns) == original_cols


def test_derived_features_dtype_float32():
    df = _make_full_nwp_df()
    out = add_derived_nwp_features(df)
    for col in DERIVED_NWP_FEATURES:
        assert out[col].dtype == np.float32, f"{col} should be float32"


def test_derived_features_missing_columns_fills_zero():
    """If the NWP df is missing columns, derived features should be 0, not crash."""
    df = _make_nwp_df()   # only nwp_var_* — none of the real columns
    out = add_derived_nwp_features(df, fill_value=0.0)
    for col in DERIVED_NWP_FEATURES:
        assert col in out.columns
        assert (out[col] == 0.0).all(), f"{col} should be 0 when source column absent"


# ── add_station_static_features ───────────────────────────────────────────────

def test_static_features_column_names():
    df = _make_nwp_df()
    meta = _make_station_metadata(["STA0", "STA1", "STA2"])
    out = add_station_static_features(df, meta)
    for col in STATIC_FEATURE_NAMES:
        assert col in out.columns, f"Missing static column: {col}"


def test_static_features_count():
    """2 continuous + 5 terrain one-hots + 4 region one-hots = 11."""
    assert len(STATIC_CONTINUOUS) == 2
    assert len(TERRAIN_COLS) == 5
    assert len(REGION_COLS) == 4
    assert len(STATIC_FEATURE_NAMES) == 11


def test_static_features_terrain_one_hot_sums_to_one():
    """Each row should have exactly one terrain_* flag set to 1."""
    df = _make_nwp_df()
    meta = _make_station_metadata(["STA0", "STA1", "STA2"])
    out = add_station_static_features(df, meta)
    row_sums = out[TERRAIN_COLS].sum(axis=1)
    assert (row_sums == 1.0).all(), "Terrain one-hot should sum to 1 per row"


def test_static_features_region_one_hot_sums_to_one():
    """Each row should have exactly one region_* flag set to 1."""
    df = _make_nwp_df()
    meta = _make_station_metadata(["STA0", "STA1", "STA2"])
    out = add_station_static_features(df, meta)
    row_sums = out[REGION_COLS].sum(axis=1)
    assert (row_sums == 1.0).all(), "Region one-hot should sum to 1 per row"


def test_static_features_terrain_ordering_deterministic():
    """TERRAIN_CLASSES and REGIONS must be alphabetically sorted (train/serve parity)."""
    assert TERRAIN_CLASSES == sorted(TERRAIN_CLASSES), "TERRAIN_CLASSES must be sorted"
    assert REGIONS == sorted(REGIONS), "REGIONS must be sorted"


def test_static_features_unknown_station_zeros():
    """Rows for stations absent from metadata should receive all-zero static features."""
    df = _make_nwp_df(n_stations=2)   # STA0, STA1
    meta = _make_station_metadata(["STA0"])   # STA1 missing
    out = add_station_static_features(df, meta)
    sta1_rows = out[out["station"] == "STA1"][STATIC_FEATURE_NAMES]
    assert (sta1_rows == 0.0).all().all(), "Unknown station should get all-zero static features"


def test_static_features_elevation_continuous():
    """elevation_m and coast_km should be numeric, not one-hot."""
    df = _make_nwp_df(n_stations=1)
    meta = {"STA0": {"elevation_m": 123.4, "coast_km": 56.7,
                     "terrain_class": "plain", "region": "central"}}
    out = add_station_static_features(df, meta)
    assert abs(out["elevation_m"].iloc[0] - 123.4) < 1e-4
    assert abs(out["coast_km"].iloc[0] - 56.7) < 1e-4


def test_static_features_does_not_mutate_input():
    df = _make_nwp_df()
    original_cols = set(df.columns)
    meta = _make_station_metadata(["STA0", "STA1", "STA2"])
    add_station_static_features(df, meta)
    assert set(df.columns) == original_cols


# ── LOCAL_STATION_IN constant ─────────────────────────────────────────────────

def test_local_station_in_is_39():
    """LOCAL_STATION_IN must match the GNN's expected input dimension."""
    assert LOCAL_STATION_IN == 39


# ── FeatureScaler ─────────────────────────────────────────────────────────────

def test_scaler_fit_transform_zero_mean():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    scaler = FeatureScaler()
    out = scaler.fit_transform(df, ["a", "b"])
    assert abs(out["a"].mean()) < 1e-6
    assert abs(out["b"].mean()) < 1e-6


def test_scaler_fit_transform_unit_variance():
    rng = np.random.default_rng(42)
    vals = rng.normal(5.0, 2.0, size=1000)
    df = pd.DataFrame({"x": vals})
    scaler = FeatureScaler()
    out = scaler.fit_transform(df, ["x"])
    assert abs(out["x"].std() - 1.0) < 0.05


def test_scaler_transform_uses_training_stats():
    """Transform on new data uses stats from fit, not new data stats."""
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    test  = pd.DataFrame({"x": [100.0, 200.0]})
    scaler = FeatureScaler()
    scaler.fit(train, ["x"])
    out = scaler.transform(test, ["x"])
    # mean from train = 20, std from train = 10 → (100-20)/10 = 8
    assert abs(out["x"].iloc[0] - 8.0) < 1e-5


def test_scaler_raises_if_not_fitted():
    scaler = FeatureScaler()
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform(df, ["x"])


def test_scaler_save_load_roundtrip():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 2.0, 3.0]})
    scaler = FeatureScaler()
    orig = scaler.fit_transform(df, ["a", "b"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scaler.npz"
        scaler.save(path)
        loaded = FeatureScaler.load(path)

    out = loaded.transform(df, ["a", "b"])
    np.testing.assert_allclose(out["a"].values, orig["a"].values, rtol=1e-5)
    np.testing.assert_allclose(out["b"].values, orig["b"].values, rtol=1e-5)


def test_scaler_constant_column_does_not_crash():
    """A column with zero variance should not raise (std floored at 1e-8)."""
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0]})
    scaler = FeatureScaler()
    out = scaler.fit_transform(df, ["const"])
    # All scaled values should be 0.0 (mean=5, std≈1e-8)
    assert (out["const"] == 0.0).all()


def test_era5_feature_names_length():
    """ERA5_FEATURE_NAMES must be exactly 11 (7 surface + 4 temporal)."""
    assert len(ERA5_SURFACE_VARS) == 7
    assert len(TEMPORAL_FEATURE_NAMES) == 4
    assert len(ERA5_FEATURE_NAMES) == 11
