"""Feature engineering for CropOS: temporal encoding and per-feature scaling.

This module is the single authoritative path from raw downloaded parquets to
the tensors the GNN consumes.  train.py, dataset.py, and the inference API all
call the same functions so train/serve features are always identical.

Feature dimensions (kept in sync with configs/model.yaml):

  ERA5 nodes   : 7 surface vars + 4 temporal  =  11 total  (era5_in = 11)

  Station nodes (local_station_in = 39):
    22  raw NWP vars
     6  derived time-varying (pressure tendency × 2, lagged precip × 2,
                               dewpoint depression, wind shear)
     2  static continuous (elevation_m, coast_km)
     5  terrain one-hot  (coastal, mountain, plain, urban, valley)
     4  region  one-hot  (central, north, northeast, south)
    ─────
    39  total

  Farm nodes   : 1 zero placeholder (farm_proj input)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── ERA5 column constants ─────────────────────────────────────────────────────

# Raw ERA5 surface variables from Open-Meteo archive API
ERA5_SURFACE_VARS: List[str] = [
    "temperature_2m",
    "dewpoint_2m",
    "relativehumidity_2m",
    "precipitation",
    "windspeed_10m",
    "winddirection_10m",
    "surface_pressure",
]

# Temporal encoding columns appended to ERA5 nodes
TEMPORAL_FEATURE_NAMES: List[str] = [
    "sin_hour",   # diurnal cycle  (period = 24 h)
    "cos_hour",
    "sin_doy",    # seasonal cycle (period = 365.25 days)
    "cos_doy",
]

# Full ordered ERA5 feature list fed to the GNN (era5_in = 11)
ERA5_FEATURE_NAMES: List[str] = ERA5_SURFACE_VARS + TEMPORAL_FEATURE_NAMES


# ── NWP feature constants ─────────────────────────────────────────────────────

# Derived features computed from raw NWP columns (all prefixed nwp_)
DERIVED_NWP_FEATURES: List[str] = [
    "nwp_dp_3h",               # surface pressure tendency over 3 h (falling = rain signal)
    "nwp_dp_6h",               # surface pressure tendency over 6 h
    "nwp_precip_3h_lag",       # precipitation 3 h ago (convective memory)
    "nwp_precip_24h_sum",      # rolling 24 h precipitation sum (soil saturation proxy)
    "nwp_dd",                  # dewpoint depression T2m − Td2m (≈0 = saturated BL)
    "nwp_wspd_shear_850_10m",  # wind speed shear 850 hPa − 10 m (low-level jet signal)
]

# Static station feature columns — same for every timestamp per station.
# Produced by add_station_static_features() from STATION_METADATA.
TERRAIN_CLASSES: List[str] = sorted(["coastal", "mountain", "plain", "urban", "valley"])
REGIONS:         List[str] = sorted(["central", "north", "northeast", "south"])

STATIC_CONTINUOUS: List[str] = ["elevation_m", "coast_km"]
TERRAIN_COLS:      List[str] = [f"terrain_{t}" for t in TERRAIN_CLASSES]
REGION_COLS:       List[str] = [f"region_{r}"  for r in REGIONS]
STATIC_FEATURE_NAMES: List[str] = STATIC_CONTINUOUS + TERRAIN_COLS + REGION_COLS  # 11 cols

# Total local_station feature dimension: 22 raw + 6 derived + 11 static = 39
LOCAL_STATION_IN: int = 39


# ── Temporal features ─────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append sin/cos-encoded temporal features from the 'timestamp' column.

    Encoding rationale: cyclic sin/cos encoding ensures the model sees
    23:00 and 01:00 as close (they are) without the arbitrary discontinuity
    of raw hour integers.

    Args:
        df: DataFrame with a 'timestamp' column (any timezone or naive).

    Returns:
        A copy of df with four additional columns:
          sin_hour, cos_hour  — diurnal cycle
          sin_doy,  cos_doy   — seasonal cycle
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.day_of_year + hour / 24.0

    df["sin_hour"] = np.sin(2 * math.pi * hour / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2 * math.pi * hour / 24.0).astype(np.float32)
    df["sin_doy"]  = np.sin(2 * math.pi * doy  / 365.25).astype(np.float32)
    df["cos_doy"]  = np.cos(2 * math.pi * doy  / 365.25).astype(np.float32)
    return df


# ── Derived NWP features ──────────────────────────────────────────────────────

def add_derived_nwp_features(
    nwp_df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Compute derived time-varying features from raw NWP columns.

    All six derived features are computed per station (grouped by 'station')
    so boundaries between stations do not bleed into each other's lag/diff.
    NaN values produced by .diff()/.shift() at the start of each station's
    series are replaced with fill_value (default 0.0).

    Args:
        nwp_df:     NWP DataFrame with a 'station' column, a 'timestamp' column,
                    and the raw nwp_* feature columns.  Need not be sorted.
        fill_value: Value to replace NaN from lag/diff at series start.

    Returns:
        A copy of nwp_df sorted by (station, timestamp) with six additional columns.
        Existing rows and all original columns are preserved.
    """
    df = nwp_df.sort_values(["station", "timestamp"]).copy()

    # ── dewpoint depression ────────────────────────────────────────────────────
    # Near zero → boundary layer near saturation → convection favoured
    t_col  = "nwp_temperature_2m"
    td_col = "nwp_dewpoint_2m"
    if t_col in df.columns and td_col in df.columns:
        df["nwp_dd"] = (
            pd.to_numeric(df[t_col], errors="coerce")
            - pd.to_numeric(df[td_col], errors="coerce")
        )
    else:
        logger.warning(
            "Dewpoint depression (nwp_dd) skipped — "
            f"missing {[c for c in [t_col, td_col] if c not in df.columns]}"
        )
        df["nwp_dd"] = fill_value

    # ── wind shear 850 hPa − 10 m ─────────────────────────────────────────────
    # Large positive value → strong low-level jet or monsoon boundary layer
    ws850 = "nwp_windspeed_850hPa"
    ws10m = "nwp_windspeed_10m"
    if ws850 in df.columns and ws10m in df.columns:
        df["nwp_wspd_shear_850_10m"] = (
            pd.to_numeric(df[ws850], errors="coerce")
            - pd.to_numeric(df[ws10m], errors="coerce")
        )
    else:
        logger.warning(
            "Wind shear (nwp_wspd_shear_850_10m) skipped — "
            f"missing {[c for c in [ws850, ws10m] if c not in df.columns]}"
        )
        df["nwp_wspd_shear_850_10m"] = fill_value

    # ── pressure tendency ─────────────────────────────────────────────────────
    # Falling pressure (negative tendency) is a strong pre-rain signal.
    # .diff(n) computes row[i] - row[i-n] within each station group.
    pres_col = "nwp_surface_pressure"
    if pres_col in df.columns:
        pres = df.groupby("station", sort=False)[pres_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce")
        )
        df["nwp_dp_3h"] = df.groupby("station", sort=False)[pres_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce").diff(3)
        )
        df["nwp_dp_6h"] = df.groupby("station", sort=False)[pres_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce").diff(6)
        )
    else:
        logger.warning(f"Pressure tendency skipped — '{pres_col}' not in NWP df")
        df["nwp_dp_3h"] = fill_value
        df["nwp_dp_6h"] = fill_value

    # ── lagged precipitation ──────────────────────────────────────────────────
    # Soil saturation and recent convective history both modulate rain risk.
    prec_col = "nwp_precipitation"
    if prec_col in df.columns:
        prec = df.groupby("station", sort=False)[prec_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce")
        )
        # 3 h lag: precipitation at t−3 (not t, to avoid leaking future signal)
        df["nwp_precip_3h_lag"] = df.groupby("station", sort=False)[prec_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce").shift(3)
        )
        # 24 h rolling sum, offset by 1 so t is not included in its own label
        df["nwp_precip_24h_sum"] = df.groupby("station", sort=False)[prec_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce")
            .shift(1)
            .rolling(24, min_periods=1)
            .sum()
        )
    else:
        logger.warning(f"Lagged precipitation skipped — '{prec_col}' not in NWP df")
        df["nwp_precip_3h_lag"] = fill_value
        df["nwp_precip_24h_sum"] = fill_value

    # Fill NaN from diffs/shifts at series boundaries
    for col in DERIVED_NWP_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value).astype(np.float32)

    return df


def add_station_static_features(
    nwp_df: pd.DataFrame,
    station_metadata: Dict[str, dict],
) -> pd.DataFrame:
    """Join static station metadata and one-hot encode categorical fields.

    Adds STATIC_FEATURE_NAMES (11 columns) to nwp_df by merging on 'station'.
    The one-hot encoding uses TERRAIN_CLASSES and REGIONS — both are sorted
    alphabetically and fixed here so train/serve output is always identical
    regardless of which stations appear in a given batch.

    Args:
        nwp_df:           NWP DataFrame with a 'station' column.
        station_metadata: {station_id: {"elevation_m": ..., "coast_km": ...,
                           "terrain_class": ..., "region": ...}}

    Returns:
        A copy of nwp_df with 11 additional static feature columns.
    """
    meta_rows = []
    for station, meta in station_metadata.items():
        meta_rows.append({
            "station":       station,
            "elevation_m":   float(meta.get("elevation_m", 0.0)),
            "coast_km":      float(meta.get("coast_km", 0.0)),
            "terrain_class": meta.get("terrain_class", "plain"),
            "region":        meta.get("region", "central"),
        })
    meta_df = pd.DataFrame(meta_rows)

    # One-hot encode with explicit, fixed categories so ordering is deterministic
    for tc in TERRAIN_CLASSES:
        meta_df[f"terrain_{tc}"] = (meta_df["terrain_class"] == tc).astype(np.float32)
    for rg in REGIONS:
        meta_df[f"region_{rg}"] = (meta_df["region"] == rg).astype(np.float32)

    keep_cols = ["station"] + STATIC_FEATURE_NAMES
    meta_df = meta_df[keep_cols]

    df = nwp_df.merge(meta_df, on="station", how="left")

    # Stations missing from metadata get zeros
    for col in STATIC_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(np.float32)

    return df


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_era5_features(
    era5_df: pd.DataFrame,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Append temporal encoding to ERA5 surface variables.

    Args:
        era5_df:   Raw ERA5 DataFrame from fetch_era5_grid().
                   Required columns: 'timestamp', 'lat', 'lon' + variables.
        variables: ERA5 variable names to include.
                   Defaults to ERA5_SURFACE_VARS (7 surface vars).

    Returns:
        DataFrame with columns: lat, lon, timestamp, <variables>,
        sin_hour, cos_hour, sin_doy, cos_doy.
        Total feature columns: len(variables) + 4 = 11 by default.
    """
    if variables is None:
        variables = ERA5_SURFACE_VARS

    missing = [v for v in variables if v not in era5_df.columns]
    if missing:
        raise ValueError(
            f"ERA5 df is missing expected variable columns: {missing}\n"
            f"Available columns: {era5_df.columns.tolist()}"
        )

    df = era5_df[["lat", "lon", "timestamp"] + variables].copy()
    df = add_temporal_features(df)
    return df


def prepare_nwp_features(
    nwp_df: pd.DataFrame,
    nwp_variables: Optional[List[str]] = None,
    station_metadata: Optional[Dict[str, dict]] = None,
    add_derived: bool = True,
    add_static: bool = True,
) -> pd.DataFrame:
    """Build the full 39-feature NWP station feature matrix.

    Processing order (all steps are applied to a copy, never mutating the input):
      1. Select the 22 raw nwp_* columns (or the subset named in nwp_variables).
      2. Compute 6 derived time-varying features (pressure tendency, lagged
         precip, dewpoint depression, wind shear) — requires time ordering per station.
      3. Join 11 static station features (elevation, coast distance, terrain/region
         one-hots) from station_metadata.

    Args:
        nwp_df:           Raw NWP DataFrame from fetch_all_stations().
                          Must have: 'station', 'timestamp', and nwp_* columns.
        nwp_variables:    Base variable names (without 'nwp_' prefix).
                          Defaults to all nwp_* columns found in nwp_df.
        station_metadata: Passed to add_station_static_features().
                          If None, static features are omitted (11 zeros per station).
        add_derived:      Compute derived time-varying features (default True).
        add_static:       Join static station features (default True).

    Returns:
        DataFrame ready to be consumed by FeatureScaler and CropOSDataset.
        Feature columns in deterministic order:
          raw nwp_* cols  |  derived nwp_* cols  |  static cols
    """
    available_nwp = [c for c in nwp_df.columns if c.startswith("nwp_")]

    if nwp_variables is not None:
        expected = [f"nwp_{v}" for v in nwp_variables]
        missing = [c for c in expected if c not in nwp_df.columns]
        if missing:
            raise ValueError(
                f"NWP df missing expected columns: {missing}\n"
                f"Available nwp_ columns: {available_nwp}"
            )
        raw_nwp_cols = [c for c in expected if c in nwp_df.columns]
    else:
        raw_nwp_cols = available_nwp

    meta_cols = [c for c in ["station", "timestamp", "lat", "lon"] if c in nwp_df.columns]
    df = nwp_df[meta_cols + raw_nwp_cols].copy()

    if add_derived:
        # add_derived_nwp_features sorts by (station, timestamp) internally
        df = add_derived_nwp_features(df)

    if add_static:
        if station_metadata is None:
            # Import the built-in metadata rather than silently emitting zeros
            try:
                from src.ingestion.metar import STATION_METADATA
                station_metadata = STATION_METADATA
                logger.info("Using built-in STATION_METADATA for static features")
            except ImportError:
                logger.warning(
                    "station_metadata=None and src.ingestion.metar not importable; "
                    "static station features will be all zeros"
                )
        if station_metadata is not None:
            df = add_station_static_features(df, station_metadata)
        else:
            # Fill zeros for all static columns so downstream code sees 39 features
            for col in STATIC_FEATURE_NAMES:
                df[col] = np.float32(0.0)

    return df


# ── Feature scaler ────────────────────────────────────────────────────────────

class FeatureScaler:
    """Per-feature StandardScaler with save/load for inference parity.

    Stores one (mean, std) pair per named feature column.  std is floored at
    1e-8 to avoid division by zero on constant-value columns (e.g. one-hot
    columns that are 1 for every station in the dataset).

    The scaler is fitted on training data ONLY and applied identically at
    inference, which is why save/load exist.  Passing column names explicitly
    (rather than fitting all columns at once) prevents accidental scaling of
    identity columns like 'lat', 'lon', or 'station'.

    Note on one-hot features: StandardScaling binary columns is technically
    correct (it normalises by frequency) but optional — the GNN's layer-norm
    means the scale matters less than for linear models.

    Usage::

        scaler = FeatureScaler()
        train_scaled = scaler.fit_transform(train_df, ERA5_FEATURE_NAMES)
        val_scaled   = scaler.transform(val_df, ERA5_FEATURE_NAMES)
        scaler.save("checkpoints/era5_scaler.npz")

        # At inference time:
        scaler = FeatureScaler.load("checkpoints/era5_scaler.npz")
        scaled = scaler.transform(live_df, ERA5_FEATURE_NAMES)
    """

    def __init__(self) -> None:
        self._means: dict[str, float] = {}
        self._stds:  dict[str, float] = {}

    # -- core API --

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "FeatureScaler":
        """Compute per-column mean and std from df (training data only)."""
        for col in feature_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            self._means[col] = float(vals.mean())
            self._stds[col]  = float(max(float(vals.std()), 1e-8))
        logger.info(f"FeatureScaler fitted on {len(feature_cols)} columns")
        return self

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Return a copy of df with feature_cols standardised.

        Raises ValueError if any column was not included in the last fit() call.
        """
        unfitted = [c for c in feature_cols if c not in self._means]
        if unfitted:
            raise ValueError(
                f"FeatureScaler not fitted for columns: {unfitted}. "
                f"Call fit() on training data first."
            )
        df = df.copy()
        for col in feature_cols:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce") - self._means[col]
            ) / self._stds[col]
        return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit on df and return the standardised copy in one step."""
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols)

    # -- persistence --

    def save(self, path: str | Path) -> None:
        """Save fitted parameters to a compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            columns=np.array(list(self._means.keys())),
            means=np.array(list(self._means.values()), dtype=np.float64),
            stds =np.array(list(self._stds.values()),  dtype=np.float64),
        )
        logger.info(f"FeatureScaler saved → {path}  ({len(self._means)} features)")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        """Load a previously saved FeatureScaler from an .npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        scaler = cls()
        scaler._means = {k: float(v) for k, v in zip(data["columns"], data["means"])}
        scaler._stds  = {k: float(v) for k, v in zip(data["columns"], data["stds"])}
        logger.info(f"FeatureScaler loaded ← {path}  ({len(scaler._means)} features)")
        return scaler

    # -- introspection --

    def feature_stats(self) -> pd.DataFrame:
        """Return a DataFrame of fitted (mean, std) values for inspection."""
        return pd.DataFrame({
            "feature": list(self._means.keys()),
            "mean":    list(self._means.values()),
            "std":     list(self._stds.values()),
        })
