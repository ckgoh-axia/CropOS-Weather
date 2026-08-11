"""CropOS PyTorch Dataset — builds per-timestamp HeteroData graphs for training.

Memory design
─────────────
ERA5 has 1,980 grid points × up to 70 k hours = ~138 M rows for Thailand.
Loading this entirely into RAM on a 7 GB GitHub Actions runner is infeasible.

The dataset therefore filters ERA5 to only those grid points within
``era5_node_radius_km`` of any METAR station before loading.  With the default
100 km radius this typically keeps 100–200 unique grid points, reducing the
in-memory ERA5 footprint to ~500 MB instead of ~10 GB.

Graph structure per timestamp
─────────────────────────────
  era5 nodes  : filtered atmospheric grid (11 features: 7 ERA5 + 4 temporal)
  metar nodes : actual airport weather observations at the 16 METAR stations (9 features)
  farm nodes  : prediction targets at the same 16 METAR locations (1 zero feature)

Edges are computed ONCE from the fixed geography and reused for every sample.

Labels
──────
  data["farm"].y  shape (n_stations, n_horizons) — binary rain (0/1) at t+h.
  Source: ERA5 precipitation at the grid point nearest each METAR station.
  Missing labels are set to 0 and a separate mask can be derived from the
  presence/absence of data in the raw DataFrames.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.features.engineer import (
    prepare_era5_features,
)
from src.features.graph_builder import build_heterogeneous_graph, haversine_km

logger = logging.getLogger(__name__)

# METAR features available from real-time Iowa State ASOS feed.
# These are the ONLY features used as local_station (metar node) inputs.
# Order is fixed — do not reorder without updating model.yaml metar_in and checkpoints.
METAR_FEATURE_COLS: List[str] = [
    "precip_mm",   # hourly precipitation (mm); 0 when not reported
    "rain_event",  # bool → float: 1.0 if RA/TS/SH wx code present
    "tmpf",        # air temperature (°F)
    "dwpf",        # dew point temperature (°F)
    "relh",        # relative humidity (%)
    "drct",        # wind direction (°)
    "sknt",        # wind speed (knots)
    "alti",        # altimeter setting (inHg)
    "vsby",        # visibility (miles)
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _nearest_era5_point(
    lat: float, lon: float, era5_lats: np.ndarray, era5_lons: np.ndarray
) -> int:
    """Return the index of the ERA5 grid point nearest to (lat, lon)."""
    if era5_lats.size == 0:
        raise ValueError(
            f"ERA5 grid is empty — no grid points within radius for station at "
            f"({lat:.3f}, {lon:.3f}). This usually means era5_recent.parquet is missing "
            f"for the requested date range. Check that era5_recent.parquet exists on HF "
            f"and covers the validation period."
        )
    dists = ((era5_lats - lat) ** 2 + (era5_lons - lon) ** 2)
    return int(np.argmin(dists))


def _filter_era5_by_radius(
    era5_df: pd.DataFrame,
    station_coords: Dict[str, Tuple[float, float]],
    radius_km: float,
) -> pd.DataFrame:
    """Keep only ERA5 rows whose (lat, lon) is within radius_km of any station."""
    unique_pts = era5_df[["lat", "lon"]].drop_duplicates()
    keep_mask = pd.Series(False, index=unique_pts.index)
    for _, (s_lat, s_lon) in station_coords.items():
        d = unique_pts.apply(
            lambda r, _lat=s_lat, _lon=s_lon: haversine_km(r["lat"], r["lon"], _lat, _lon), axis=1
        )
        keep_mask |= d <= radius_km

    keep_pts = unique_pts[keep_mask].set_index(["lat", "lon"])
    filtered = era5_df.set_index(["lat", "lon"])
    filtered = filtered[filtered.index.isin(keep_pts.index)].reset_index()
    logger.info(
        f"ERA5 filtered: {keep_mask.sum()} / {len(unique_pts)} grid points "
        f"within {radius_km} km of any station"
    )
    return filtered


def _build_era5_label_df(
    era5_df: pd.DataFrame,
    station_coords: Dict[str, Tuple[float, float]],
    station_order: List[str],
) -> pd.DataFrame:
    """Build per-station ERA5 precipitation labels from nearest grid points.

    Args:
        era5_df:       Full or filtered ERA5 DataFrame with 'precipitation' column.
        station_coords:{station_id: (lat, lon)}
        station_order: Ordered list of station IDs (defines farm node ordering).

    Returns:
        DataFrame with columns: timestamp, station, precip_mm.
        One row per (timestamp, station) pair.
    """
    unique_pts = era5_df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    era5_lats = unique_pts["lat"].values
    era5_lons = unique_pts["lon"].values

    # Map each station to its nearest ERA5 grid point
    station_to_gridpt: dict[str, tuple[float, float]] = {}
    for station in station_order:
        s_lat, s_lon = station_coords[station]
        idx = _nearest_era5_point(s_lat, s_lon, era5_lats, era5_lons)
        station_to_gridpt[station] = (float(era5_lats[idx]), float(era5_lons[idx]))

    # Determine precipitation column name
    if "precipitation" in era5_df.columns:
        precip_col = "precipitation"
    elif "precipitation_sum" in era5_df.columns:
        precip_col = "precipitation_sum"
    else:
        raise ValueError(
            f"No precipitation column in ERA5 df. Columns: {era5_df.columns.tolist()}"
        )

    frames = []
    for station, (g_lat, g_lon) in station_to_gridpt.items():
        mask = (era5_df["lat"].round(4) == round(g_lat, 4)) & \
               (era5_df["lon"].round(4) == round(g_lon, 4))
        sub = era5_df[mask][["timestamp", precip_col]].copy()
        sub = sub.rename(columns={precip_col: "precip_mm"})
        sub["station"] = station
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "station", "precip_mm"])

    result = pd.concat(frames, ignore_index=True)
    result["precip_mm"] = pd.to_numeric(result["precip_mm"], errors="coerce").fillna(0.0)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result[["timestamp", "station", "precip_mm"]]


# ── dataset ───────────────────────────────────────────────────────────────────

class CropOSDataset(Dataset):
    """Per-timestamp HeteroData graphs for training CropOSGNN.

    Each sample corresponds to one UTC hour and contains:
      - era5 nodes  : atmospheric features for ERA5 grid points near stations
      - metar nodes : actual METAR observations at the 16 airport locations (9 features)
      - farm nodes  : zero-feature placeholders at the same 16 locations
      - farm labels : data["farm"].y of shape (n_stations, n_horizons)

    Args:
        era5_df:        ERA5 DataFrame (will be filtered to era5_node_radius_km).
                        Must have: timestamp, lat, lon + ERA5_SURFACE_VARS.
        metar_df:       METAR DataFrame from fetch_all_thai_stations().
                        Must have: timestamp, station + METAR_FEATURE_COLS.
        station_order:  Ordered list of ICAO station IDs defining node ordering.
        station_coords: {station_id: (lat, lon)} for all stations in station_order.
        era5_vars:      ERA5 surface variable names (default: ERA5_SURFACE_VARS).
        horizons_h:     Forecast horizons in hours (default: [12, 24, 36, 48]).
        era5_node_radius_km: ERA5 nodes included if within this radius of any station.
        threshold_mm:   Precipitation threshold for rain/no-rain label (default 1.0 mm).
    """

    def __init__(
        self,
        era5_df: pd.DataFrame,
        metar_df: pd.DataFrame,
        station_order: List[str],
        station_coords: Dict[str, Tuple[float, float]],
        era5_vars: List[str] | None = None,
        horizons_h: List[int] | None = None,
        era5_node_radius_km: float = 100.0,
        threshold_mm: float = 1.0,
    ) -> None:
        super().__init__()

        if era5_vars is None:
            from src.features.engineer import ERA5_SURFACE_VARS
            era5_vars = ERA5_SURFACE_VARS
        if horizons_h is None:
            horizons_h = [12, 24, 36, 48]

        self.station_order = station_order
        self.station_coords = station_coords
        self.horizons_h = horizons_h
        self.n_stations = len(station_order)
        self.n_horizons = len(horizons_h)
        self.threshold_mm = threshold_mm

        # ── 1. filter ERA5 to manageable node set ──────────────────────────
        print("[DS] Filtering ERA5 to nodes near stations...", flush=True)
        era5_df = _filter_era5_by_radius(era5_df, station_coords, era5_node_radius_km)

        # ── 2. add temporal features to ERA5 ──────────────────────────────
        print("[DS] Adding temporal features to ERA5...", flush=True)
        era5_df = prepare_era5_features(era5_df, variables=era5_vars)
        era5_feature_cols = era5_vars + ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
        self.era5_feature_cols = era5_feature_cols
        self.n_era5_features = len(era5_feature_cols)

        # ── 3. METAR feature columns (fixed set, same as real-time prod feed) ─
        self.metar_feature_cols = METAR_FEATURE_COLS
        self.n_metar_features = len(METAR_FEATURE_COLS)

        # ── 4. unique sorted ERA5 grid points (fixed geography) ────────────
        unique_pts = (
            era5_df[["lat", "lon"]]
            .drop_duplicates()
            .sort_values(["lat", "lon"])
            .reset_index(drop=True)
        )
        self.n_era5_nodes = len(unique_pts)
        self._era5_lat = unique_pts["lat"].values
        self._era5_lon = unique_pts["lon"].values

        # ── 5. build ERA5 lookup dict {timestamp → (n_era5_nodes, n_feat)} ─
        # Sort by (timestamp, lat, lon) so within-group row order matches
        # unique_pts (which is also sorted by lat, lon). This makes the numpy
        # array at each timestamp consistently index-aligned with _era5_lat/_era5_lon.
        print("[DS] Building ERA5 timestamp lookup (this may take ~30 s)...", flush=True)
        era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"], utc=True)
        era5_df = era5_df.sort_values(["timestamp", "lat", "lon"])

        self._era5_by_ts: dict[pd.Timestamp, np.ndarray] = {}
        for ts, grp in era5_df.groupby("timestamp"):
            self._era5_by_ts[ts] = (
                grp[era5_feature_cols].values.astype(np.float32)
            )
        _era5_keys = list(self._era5_by_ts.keys())
        print(
            f"[DS] ERA5 lookup: {len(self._era5_by_ts):,} timestamps | "
            f"range: {min(_era5_keys) if _era5_keys else 'EMPTY'} → "
            f"{max(_era5_keys) if _era5_keys else 'EMPTY'}",
            flush=True,
        )

        # ── 6. build METAR lookup dict {timestamp → (n_stations, n_metar_feat)} ─
        print("[DS] Building METAR timestamp lookup...", flush=True)
        metar_df = metar_df.copy()
        metar_df["timestamp"] = pd.to_datetime(metar_df["timestamp"], utc=True)

        # Cast rain_event (bool) to float so numpy stores it correctly
        if "rain_event" in metar_df.columns:
            metar_df["rain_event"] = metar_df["rain_event"].astype(float)

        # Ensure all METAR feature columns are numeric
        for col in METAR_FEATURE_COLS:
            if col in metar_df.columns and col != "rain_event":
                metar_df[col] = pd.to_numeric(metar_df[col], errors="coerce")

        self._metar_by_ts: dict[pd.Timestamp, np.ndarray] = {}
        for ts, grp in metar_df.groupby("timestamp"):
            arr = (
                grp.set_index("station")
                .reindex(station_order)[METAR_FEATURE_COLS]
                .values.astype(np.float32)
            )
            # NaN → 0: missing observation treated as sensor-absent (same as DropNode)
            self._metar_by_ts[ts] = np.nan_to_num(arr, nan=0.0)
        _metar_keys = list(self._metar_by_ts.keys())
        print(
            f"[DS] METAR lookup: {len(self._metar_by_ts):,} timestamps | "
            f"range: {min(_metar_keys) if _metar_keys else 'EMPTY'} → "
            f"{max(_metar_keys) if _metar_keys else 'EMPTY'}",
            flush=True,
        )

        # ── METAR data-quality check ─────────────────────────────────────────
        if self._metar_by_ts:
            _sample_keys = list(self._metar_by_ts.keys())[:min(100, len(self._metar_by_ts))]
            _sample_arr = np.concatenate([self._metar_by_ts[k] for k in _sample_keys], axis=0)
            _zero_frac = float((_sample_arr == 0.0).mean())
            _std = float(np.nanstd(_sample_arr))
            logger.info(
                f"METAR quality (first {len(_sample_keys)} ts, pre-scale): "
                f"zero_frac={_zero_frac:.3f}  "
                f"mean={float(np.nanmean(_sample_arr)):.4f}  std={_std:.4f}"
            )
            if _zero_frac > 0.60:
                logger.warning(
                    f"METAR zero_frac={_zero_frac:.1%} — likely widespread missing observations. "
                    f"Check metar_thai.parquet coverage for the requested date range."
                )

        del metar_df  # free memory — all data is in _metar_by_ts

        # ── 7. build label array [n_ts, n_stations, n_horizons] ────────────
        print("[DS] Building ERA5 precipitation label table...", flush=True)
        label_df = _build_era5_label_df(
            era5_df, station_coords, station_order
        )
        label_df = label_df.set_index(["timestamp", "station"])
        del era5_df  # free the filtered ERA5 DataFrame — all data now in _era5_by_ts + label_df
        import gc
        gc.collect()

        # Timestamps where both ERA5 and METAR data are available
        common_ts = sorted(
            set(self._era5_by_ts) & set(self._metar_by_ts),
            key=lambda t: t,
        )
        print(
            f"[DS] ERA5∩METAR intersection: {len(common_ts):,} timestamps | "
            f"range: {common_ts[0] if common_ts else 'EMPTY'} → "
            f"{common_ts[-1] if common_ts else 'EMPTY'}",
            flush=True,
        )
        if len(common_ts) == 0:
            print(
                f"[DS] DIAGNOSIS: Intersection is EMPTY.\n"
                f"     ERA5 keys: {len(self._era5_by_ts):,}  "
                f"(sample: {list(self._era5_by_ts.keys())[:3]})\n"
                f"     METAR keys: {len(self._metar_by_ts):,}  "
                f"(sample: {list(self._metar_by_ts.keys())[:3]})\n"
                f"     Possible causes:\n"
                f"     1) era5_recent.parquet does not cover {horizons_h} — "
                f"check its max timestamp above.\n"
                f"     2) metar_thai.parquet was downloaded with an old end_date — "
                f"check its max timestamp above.\n"
                f"     3) METAR timestamps are sub-hourly (not rounded to :00) — "
                f"compare sample keys above.",
                flush=True,
            )
        # Keep only timestamps for which ALL future label timestamps exist
        label_ts = set(label_df.index.get_level_values("timestamp"))
        valid_ts: list[pd.Timestamp] = []
        for ts in common_ts:
            if all(ts + pd.Timedelta(hours=h) in label_ts for h in horizons_h):
                valid_ts.append(ts)

        self.timestamps: list[pd.Timestamp] = valid_ts
        n_ts = len(valid_ts)
        print(f"[DS] Valid timestamps (ERA5∩METAR with full label horizon): {n_ts:,}", flush=True)

        print("[DS] Pre-computing label array...", flush=True)
        label_arr = np.zeros((n_ts, self.n_stations, self.n_horizons), dtype=np.float32)
        for h_idx, h in enumerate(horizons_h):
            h_delta = pd.Timedelta(hours=h)
            for ts_idx, ts in enumerate(valid_ts):
                future_ts = ts + h_delta
                for s_idx, station in enumerate(station_order):
                    try:
                        precip = label_df.loc[(future_ts, station), "precip_mm"]
                        label_arr[ts_idx, s_idx, h_idx] = float(
                            float(precip) >= threshold_mm
                        )
                    except KeyError:
                        pass  # remains 0; caller may apply mask if needed
        self._label_arr = label_arr

        # Continuous precipitation mm array for regression head and evaluation.
        print("[DS] Pre-computing continuous mm label array...", flush=True)
        mm_arr = np.zeros((n_ts, self.n_stations, self.n_horizons), dtype=np.float32)
        for h_idx, h in enumerate(horizons_h):
            h_delta = pd.Timedelta(hours=h)
            for ts_idx, ts in enumerate(valid_ts):
                future_ts = ts + h_delta
                for s_idx, station in enumerate(station_order):
                    try:
                        precip = label_df.loc[(future_ts, station), "precip_mm"]
                        mm_arr[ts_idx, s_idx, h_idx] = max(0.0, float(precip))
                    except KeyError:
                        pass  # remains 0 — missing label treated as no rain
        self._mm_arr = mm_arr

        if label_arr.size == 0:
            logger.warning(
                "EMPTY DATASET: no valid timestamps found for this split. "
                "ERA5 and METAR timestamps do not overlap, or all label "
                "horizon timestamps are missing. Training will be a no-op."
            )
        else:
            logger.info(
                f"Rain fraction across labels: "
                f"{float(label_arr.mean()):.3f}  "
                f"(threshold={threshold_mm} mm)"
            )

        # ── 8. pre-build fixed edge tensors ───────────────────────────────
        print("[DS] Building fixed graph edges...", flush=True)
        era5_node_list = [
            {"lat": float(self._era5_lat[i]), "lon": float(self._era5_lon[i]),
             "feats": [0.0] * self.n_era5_features}
            for i in range(self.n_era5_nodes)
        ]
        metar_node_list = [
            {"lat": station_coords[s][0], "lon": station_coords[s][1],
             "feats": [0.0] * self.n_metar_features}
            for s in station_order
        ]
        farm_node_list = [
            {"lat": station_coords[s][0], "lon": station_coords[s][1]}
            for s in station_order
        ]
        base = build_heterogeneous_graph(
            era5_nodes=era5_node_list,
            metar_nodes=metar_node_list,
            farm_nodes=farm_node_list,
            edge_radius_km=era5_node_radius_km * 2.0,  # broad enough to cover all nearby ERA5
        )
        self._edge_era5_to_metar = base["era5", "to", "metar"].edge_index
        self._edge_era5_to_farm  = base["era5", "to", "farm"].edge_index
        self._edge_metar_to_farm = base["metar", "to", "farm"].edge_index
        print("[DS] Dataset ready.", flush=True)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> HeteroData:
        ts = self.timestamps[idx]

        # ERA5 node features
        era5_feats = self._era5_by_ts.get(ts)
        if era5_feats is None:
            era5_feats = np.zeros(
                (self.n_era5_nodes, self.n_era5_features), dtype=np.float32
            )

        # METAR station features
        metar_feats = self._metar_by_ts.get(ts)
        if metar_feats is None:
            metar_feats = np.zeros(
                (self.n_stations, self.n_metar_features), dtype=np.float32
            )

        # Assemble HeteroData
        data = HeteroData()
        data["era5"].x = torch.from_numpy(era5_feats)
        data["metar"].x = torch.from_numpy(metar_feats)
        data["farm"].x = torch.zeros(self.n_stations, 1, dtype=torch.float)
        data["farm"].y = torch.from_numpy(self._label_arr[idx])
        data["farm"].precip_mm = torch.from_numpy(self._mm_arr[idx])

        data["era5", "to", "metar"].edge_index = self._edge_era5_to_metar
        data["era5", "to", "farm"].edge_index = self._edge_era5_to_farm
        data["metar", "to", "farm"].edge_index = self._edge_metar_to_farm

        return data


# ── Factory / loader ──────────────────────────────────────────────────────────

def load_dataset_from_parquets(
    era5_path: str | Path,
    metar_path: str | Path,
    station_order: List[str],
    station_coords: Dict[str, Tuple[float, float]],
    start_date: str | None = None,
    end_date: str | None = None,
    era5_node_radius_km: float = 100.0,
    horizons_h: List[int] | None = None,
    threshold_mm: float = 1.0,
    era5_north_path: str | Path | None = None,
    era5_recent_path: str | Path | None = None,
) -> CropOSDataset:
    """Build a CropOSDataset from on-disk or HuggingFace-cached parquet files.

    Args:
        era5_path:        Path to era5_thailand.parquet (southern/existing grid).
        metar_path:       Path to metar_thai.parquet (actual airport observations).
        station_order:    Ordered list of METAR station IDs.
        station_coords:   {station_id: (lat, lon)}.
        start_date:       Clip to timestamps >= start_date (ISO format, UTC).
        end_date:         Clip to timestamps <= end_date (ISO format, UTC).
        era5_node_radius_km: ERA5 spatial filtering radius (default 100 km).
        horizons_h:       Forecast horizons in hours.
        threshold_mm:     Rain/no-rain threshold.
        era5_north_path:  Optional path to era5_north.parquet (northern grid top-up).
                          If provided, concatenated with era5_path before filtering.
        era5_recent_path: Optional path to era5_recent.parquet (recent timestamps top-up).
                          If provided and file exists, concatenated with era5_path.
                          Silently ignored if path does not exist.

    Returns:
        Constructed CropOSDataset ready for DataLoader.
    """
    # Build pyarrow row-group filters so we only materialise the requested date slice.
    # This dramatically cuts peak RAM when loading a multi-year parquet for a short
    # validation window (e.g. 1 year out of 9).  Falls back silently if the parquet
    # stores timestamps as strings or has no row-group statistics.
    def _pq_date_filters(start: str | None, end: str | None) -> list | None:
        filters: list[tuple] = []
        if start:
            filters.append(("timestamp", ">=", pd.Timestamp(start, tz="UTC")))
        if end:
            filters.append(("timestamp", "<=", pd.Timestamp(end, tz="UTC")))
        return filters if filters else None

    print(
        f"[DATA] Loading ERA5 from {era5_path} (filter: {start_date} – {end_date})...",
        flush=True,
    )
    # Load only the columns that CropOSDataset actually needs.
    # This cuts peak RAM from ~12 GB → ~3 GB for a multi-year ERA5 parquet
    # by avoiding materialising NWP/extended columns in memory.
    from src.features.engineer import ERA5_SURFACE_VARS as _ERA5_SURFACE_VARS
    _era5_cols = ["timestamp", "lat", "lon"] + list(_ERA5_SURFACE_VARS)
    try:
        era5_df = pd.read_parquet(
            era5_path,
            columns=_era5_cols,
            filters=_pq_date_filters(start_date, end_date),
        )
    except Exception:
        era5_df = pd.read_parquet(era5_path, columns=_era5_cols)
    era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"], utc=True)
    if start_date:
        era5_df = era5_df[era5_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        era5_df = era5_df[era5_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    _era5_ts = era5_df["timestamp"]
    print(
        f"[DATA] ERA5 base: {len(era5_df):,} rows | "
        f"ts range: {_era5_ts.min()} → {_era5_ts.max()} | "
        f"unique ts: {_era5_ts.nunique():,}",
        flush=True,
    )

    # Merge northern top-up if available (adds grid points for Bangkok, Chiang Mai, etc.)
    if era5_north_path is not None and Path(era5_north_path).exists():
        print(f"[DATA] Loading ERA5 north top-up from {era5_north_path}...", flush=True)
        try:
            north_df = pd.read_parquet(
                era5_north_path,
                columns=_era5_cols,
                filters=_pq_date_filters(start_date, end_date),
            )
        except Exception:
            north_df = pd.read_parquet(era5_north_path, columns=_era5_cols)
        north_df["timestamp"] = pd.to_datetime(north_df["timestamp"], utc=True)
        if start_date:
            north_df = north_df[north_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            north_df = north_df[north_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
        era5_df = pd.concat([era5_df, north_df], ignore_index=True)
        del north_df
        print(
            f"[DATA] ERA5 (south + north): {len(era5_df):,} rows, "
            f"{era5_df[['lat','lon']].drop_duplicates().__len__()} unique grid points",
            flush=True,
        )
    else:
        print(
            f"[DATA] ERA5: {len(era5_df):,} rows, {era5_df['timestamp'].nunique():,} timestamps",
            flush=True,
        )

    # Merge recent ERA5 top-up if available (extends training set with newer timestamps)
    if era5_recent_path is not None and Path(era5_recent_path).exists():
        print(f"[DATA] Loading ERA5 recent from {era5_recent_path}...", flush=True)
        try:
            recent_era5_df = pd.read_parquet(
                era5_recent_path,
                columns=_era5_cols,
                filters=_pq_date_filters(start_date, end_date),
            )
        except Exception:
            recent_era5_df = pd.read_parquet(era5_recent_path, columns=_era5_cols)
        recent_era5_df["timestamp"] = pd.to_datetime(recent_era5_df["timestamp"], utc=True)
        # Show full range before filtering so we can see if the file even has recent data
        _rec_min_raw = recent_era5_df["timestamp"].min() if len(recent_era5_df) > 0 else "EMPTY"
        _rec_max_raw = recent_era5_df["timestamp"].max() if len(recent_era5_df) > 0 else "EMPTY"
        print(
            f"[DATA] ERA5 recent (raw from file): {len(recent_era5_df):,} rows | "
            f"full range: {_rec_min_raw} → {_rec_max_raw}",
            flush=True,
        )
        if start_date:
            recent_era5_df = recent_era5_df[
                recent_era5_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")
            ]
        if end_date:
            recent_era5_df = recent_era5_df[
                recent_era5_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")
            ]
        _rec_min_filt = recent_era5_df["timestamp"].min() if len(recent_era5_df) > 0 else "EMPTY"
        _rec_max_filt = recent_era5_df["timestamp"].max() if len(recent_era5_df) > 0 else "EMPTY"
        print(
            f"[DATA] ERA5 recent (after {start_date}–{end_date} filter): "
            f"{len(recent_era5_df):,} rows | range: {_rec_min_filt} → {_rec_max_filt}",
            flush=True,
        )
        if len(recent_era5_df) == 0:
            print(
                f"[DATA] WARNING: era5_recent.parquet has NO rows for the requested period "
                f"({start_date} – {end_date}). Its actual data only goes up to {_rec_max_raw}. "
                f"Re-run the ERA5 Extend workflow and wait for it to reach {start_date}.",
                flush=True,
            )
        era5_df = pd.concat([era5_df, recent_era5_df], ignore_index=True)
        del recent_era5_df
        _era5_ts2 = era5_df["timestamp"]
        print(
            f"[DATA] ERA5 (base + recent): {len(era5_df):,} rows | "
            f"ts range: {_era5_ts2.min()} → {_era5_ts2.max()}",
            flush=True,
        )
    elif era5_recent_path is not None:
        print(
            f"[DATA] WARNING: era5_recent_path={era5_recent_path} not found on disk — skipping. "
            f"Validation ERA5 will likely be empty for {start_date}–{end_date}.",
            flush=True,
        )

    print(f"[DATA] Loading METAR from {metar_path}...", flush=True)
    try:
        metar_df = pd.read_parquet(metar_path, filters=_pq_date_filters(start_date, end_date))
    except Exception:
        metar_df = pd.read_parquet(metar_path)
    metar_df["timestamp"] = pd.to_datetime(metar_df["timestamp"], utc=True)
    _met_min_raw = metar_df["timestamp"].min() if len(metar_df) > 0 else "EMPTY"
    _met_max_raw = metar_df["timestamp"].max() if len(metar_df) > 0 else "EMPTY"
    print(
        f"[DATA] METAR (raw from file): {len(metar_df):,} rows | "
        f"full range: {_met_min_raw} → {_met_max_raw} | "
        f"stations: {metar_df['station'].nunique() if len(metar_df) > 0 else 0}",
        flush=True,
    )
    if start_date:
        metar_df = metar_df[metar_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        metar_df = metar_df[metar_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    _met_min_filt = metar_df["timestamp"].min() if len(metar_df) > 0 else "EMPTY"
    _met_max_filt = metar_df["timestamp"].max() if len(metar_df) > 0 else "EMPTY"
    print(
        f"[DATA] METAR (after {start_date}–{end_date} filter): "
        f"{len(metar_df):,} rows | range: {_met_min_filt} → {_met_max_filt} | "
        f"stations: {metar_df['station'].nunique() if len(metar_df) > 0 else 0}",
        flush=True,
    )
    if len(metar_df) == 0:
        print(
            f"[DATA] WARNING: metar_thai.parquet has NO rows for {start_date}–{end_date}. "
            f"Its data only goes up to {_met_max_raw}. "
            f"Re-run the Download Training Data workflow with "
            f"end_date extended to cover this period.",
            flush=True,
        )

    # Round sub-hourly METAR timestamps to the nearest hour so they align with ERA5.
    # Iowa State ASOS routine observations are at :53-:56 past the hour; ERA5 is on the hour.
    # Without rounding, the ERA5∩METAR set intersection is empty for every split.
    # If timestamps are already at :00 this is a no-op. Keep the latest obs per (hour, station)
    # so special observations don't create duplicates that break set_index("station").
    if len(metar_df) > 0:
        _sample_pre_round = metar_df["timestamp"].head(3).tolist()
        metar_df["timestamp"] = metar_df["timestamp"].dt.round("h")
        _sample_post_round = metar_df["timestamp"].head(3).tolist()
        print(
            f"[DATA] METAR timestamp rounding (pre→post hour): "
            f"{_sample_pre_round} → {_sample_post_round}",
            flush=True,
        )
        metar_df = (
            metar_df
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp", "station"], keep="last")
            .reset_index(drop=True)
        )
        print(
            f"[DATA] METAR after dedup: {len(metar_df):,} rows | "
            f"unique (ts,stn) pairs per-hour",
            flush=True,
        )

    return CropOSDataset(
        era5_df=era5_df,
        metar_df=metar_df,
        station_order=station_order,
        station_coords=station_coords,
        horizons_h=horizons_h,
        era5_node_radius_km=era5_node_radius_km,
        threshold_mm=threshold_mm,
    )


def load_dataset_from_hf(
    repo_id: str,
    hf_token: str,
    station_order: List[str],
    station_coords: Dict[str, Tuple[float, float]],
    start_date: str | None = None,
    end_date: str | None = None,
    era5_node_radius_km: float = 100.0,
    horizons_h: List[int] | None = None,
    threshold_mm: float = 1.0,
    cache_dir: str | Path | None = None,
) -> CropOSDataset:
    """Build a CropOSDataset by downloading parquets from HuggingFace.

    The downloaded files are cached locally so repeated runs do not re-download.

    Args:
        repo_id:  HuggingFace dataset repo ID (e.g. 'username/cropos-data').
        hf_token: HuggingFace API token (read access).
        (other args as in load_dataset_from_parquets)
    """
    from huggingface_hub import hf_hub_download

    def _dl(filename: str) -> Path:
        return Path(hf_hub_download(  # nosec B615 — dataset repo, no revision pinning needed; latest is always correct
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
            cache_dir=str(cache_dir) if cache_dir else None,
        ))

    era5_path = _dl("era5_thailand.parquet")

    # Download northern top-up if it exists on HF (grid points above ~12°N)
    era5_north_path: Path | None = None
    try:
        era5_north_path = _dl("era5_north.parquet")
        logger.info("Found era5_north.parquet on HF — northern grid top-up will be merged")
    except Exception:
        logger.info("era5_north.parquet not on HF yet — using southern grid only")

    # Download recent ERA5 top-up if it exists on HF (covers validation / recent years)
    era5_recent_path: Path | None = None
    try:
        era5_recent_path = _dl("era5_recent.parquet")
        logger.info("Found era5_recent.parquet on HF — recent ERA5 timestamps will be merged")
    except Exception:
        logger.info("era5_recent.parquet not on HF yet — validation may have no ERA5 rows")

    # METAR observations — these ARE the local_station (metar node) features for training.
    # metar_thai.parquet covers the full historical range; date filtering is applied in
    # load_dataset_from_parquets so the same file serves both train and val datasets.
    metar_path = _dl("metar_thai.parquet")
    logger.info("Using metar_thai.parquet for metar node features")

    # NWP forecast files — downloaded for future comparative study but NOT used in training.
    # NWP forecasts are not available in real-time production; only METAR observations are.
    for nwp_file in ["nwp_features.parquet", "nwp_recent.parquet"]:
        try:
            _dl(nwp_file)
            logger.info(f"Downloaded {nwp_file} (comparative study; not used in training)")
        except Exception:
            logger.debug(f"{nwp_file} not on HF — skipping")

    return load_dataset_from_parquets(
        era5_path=era5_path,
        metar_path=metar_path,
        station_order=station_order,
        station_coords=station_coords,
        start_date=start_date,
        end_date=end_date,
        era5_node_radius_km=era5_node_radius_km,
        horizons_h=horizons_h,
        threshold_mm=threshold_mm,
        era5_north_path=era5_north_path,
        era5_recent_path=era5_recent_path,
    )
