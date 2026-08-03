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
  era5 nodes         : filtered atmospheric grid (11 features: 7 ERA5 + 4 temporal)
  local_station nodes: GFS NWP forecasts at the 16 METAR stations (22 features)
  farm nodes         : prediction targets at the same 16 METAR locations (1 zero feature)

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _nearest_era5_point(
    lat: float, lon: float, era5_lats: np.ndarray, era5_lons: np.ndarray
) -> int:
    """Return the index of the ERA5 grid point nearest to (lat, lon)."""
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
      - era5 nodes   : atmospheric features for ERA5 grid points near stations
      - station nodes: GFS NWP features at the 16 METAR airport locations
      - farm nodes   : zero-feature placeholders at the same 16 locations
      - farm labels  : data["farm"].y of shape (n_stations, n_horizons)

    Args:
        era5_df:        ERA5 DataFrame (will be filtered to era5_node_radius_km).
                        Must have: timestamp, lat, lon + ERA5_SURFACE_VARS.
        nwp_df:         NWP DataFrame from fetch_all_stations().
                        Must have: timestamp, station, lat, lon + nwp_* columns.
        station_order:  Ordered list of ICAO station IDs defining node ordering.
        station_coords: {station_id: (lat, lon)} for all stations in station_order.
        era5_vars:      ERA5 surface variable names (default: ERA5_SURFACE_VARS).
        nwp_var_cols:   NWP feature column names with nwp_ prefix (default: all nwp_*).
        horizons_h:     Forecast horizons in hours (default: [12, 24, 36, 48]).
        era5_node_radius_km: ERA5 nodes included if within this radius of any station.
        threshold_mm:   Precipitation threshold for rain/no-rain label (default 1.0 mm).
    """

    def __init__(
        self,
        era5_df: pd.DataFrame,
        nwp_df: pd.DataFrame,
        station_order: List[str],
        station_coords: Dict[str, Tuple[float, float]],
        era5_vars: List[str] | None = None,
        nwp_var_cols: List[str] | None = None,
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
        logger.info("Filtering ERA5 to nodes near stations...")
        era5_df = _filter_era5_by_radius(era5_df, station_coords, era5_node_radius_km)

        # ── 2. add temporal features to ERA5 ──────────────────────────────
        logger.info("Adding temporal features to ERA5...")
        era5_df = prepare_era5_features(era5_df, variables=era5_vars)
        era5_feature_cols = era5_vars + ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
        self.era5_feature_cols = era5_feature_cols
        self.n_era5_features = len(era5_feature_cols)

        # ── 3. determine NWP feature columns ──────────────────────────────
        if nwp_var_cols is None:
            nwp_var_cols = [c for c in nwp_df.columns if c.startswith("nwp_")]
        self.nwp_feature_cols = nwp_var_cols
        self.n_nwp_features = len(nwp_var_cols)

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
        logger.info("Building ERA5 timestamp lookup (this may take ~30 s)...")
        era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"], utc=True)
        era5_df = era5_df.sort_values(["timestamp", "lat", "lon"])

        self._era5_by_ts: dict[pd.Timestamp, np.ndarray] = {}
        for ts, grp in era5_df.groupby("timestamp"):
            self._era5_by_ts[ts] = (
                grp[era5_feature_cols].values.astype(np.float32)
            )
        logger.info(f"ERA5 lookup built: {len(self._era5_by_ts):,} timestamps")

        # ── 6. build NWP lookup dict {timestamp → (n_stations, n_nwp_feat)} ─
        logger.info("Building NWP timestamp lookup...")
        nwp_df = nwp_df.copy()
        nwp_df["timestamp"] = pd.to_datetime(nwp_df["timestamp"], utc=True)

        self._nwp_by_ts: dict[pd.Timestamp, np.ndarray] = {}
        for ts, grp in nwp_df.groupby("timestamp"):
            arr = (
                grp.set_index("station")
                .reindex(station_order)[nwp_var_cols]
                .values.astype(np.float32)
            )
            self._nwp_by_ts[ts] = np.nan_to_num(arr, nan=0.0)
        logger.info(f"NWP lookup built: {len(self._nwp_by_ts):,} timestamps")
        del nwp_df  # free memory — all data is in _nwp_by_ts

        # ── 7. build label array [n_ts, n_stations, n_horizons] ────────────
        logger.info("Building ERA5 precipitation label table...")
        label_df = _build_era5_label_df(
            era5_df, station_coords, station_order
        )
        label_df = label_df.set_index(["timestamp", "station"])

        # Timestamps where all ERA5 and NWP data are available
        common_ts = sorted(
            set(self._era5_by_ts) & set(self._nwp_by_ts),
            key=lambda t: t,
        )
        # Keep only timestamps for which ALL future label timestamps exist
        label_ts = set(label_df.index.get_level_values("timestamp"))
        valid_ts: list[pd.Timestamp] = []
        for ts in common_ts:
            if all(ts + pd.Timedelta(hours=h) in label_ts for h in horizons_h):
                valid_ts.append(ts)

        self.timestamps: list[pd.Timestamp] = valid_ts
        n_ts = len(valid_ts)
        logger.info(f"Valid training timestamps: {n_ts:,}")

        logger.info("Pre-computing label array...")
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
        logger.info(
            f"Rain fraction across labels: "
            f"{float(label_arr.mean()):.3f}  "
            f"(threshold={threshold_mm} mm)"
        )

        # ── 8. pre-build fixed edge tensors ───────────────────────────────
        logger.info("Building fixed graph edges...")
        era5_node_list = [
            {"lat": float(self._era5_lat[i]), "lon": float(self._era5_lon[i]),
             "feats": [0.0] * self.n_era5_features}
            for i in range(self.n_era5_nodes)
        ]
        station_node_list = [
            {"lat": station_coords[s][0], "lon": station_coords[s][1],
             "feats": [0.0] * self.n_nwp_features}
            for s in station_order
        ]
        farm_node_list = [
            {"lat": station_coords[s][0], "lon": station_coords[s][1]}
            for s in station_order
        ]
        base = build_heterogeneous_graph(
            era5_nodes=era5_node_list,
            local_station_nodes=station_node_list,
            farm_nodes=farm_node_list,
            edge_radius_km=era5_node_radius_km * 2.0,  # broad enough to cover all nearby ERA5
        )
        self._edge_era5_to_local = base["era5", "to", "local_station"].edge_index
        self._edge_era5_to_farm  = base["era5", "to", "farm"].edge_index
        self._edge_local_to_farm = base["local_station", "to", "farm"].edge_index
        logger.info("Dataset ready.")

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

        # NWP station features
        nwp_feats = self._nwp_by_ts.get(ts)
        if nwp_feats is None:
            nwp_feats = np.zeros(
                (self.n_stations, self.n_nwp_features), dtype=np.float32
            )

        # Assemble HeteroData
        data = HeteroData()
        data["era5"].x = torch.from_numpy(era5_feats)
        data["local_station"].x = torch.from_numpy(nwp_feats)
        data["farm"].x = torch.zeros(self.n_stations, 1, dtype=torch.float)
        data["farm"].y = torch.from_numpy(self._label_arr[idx])

        data["era5", "to", "local_station"].edge_index = self._edge_era5_to_local
        data["era5", "to", "farm"].edge_index = self._edge_era5_to_farm
        data["local_station", "to", "farm"].edge_index = self._edge_local_to_farm

        return data


# ── Factory / loader ──────────────────────────────────────────────────────────

def load_dataset_from_parquets(
    era5_path: str | Path,
    nwp_path: str | Path,
    station_order: List[str],
    station_coords: Dict[str, Tuple[float, float]],
    start_date: str | None = None,
    end_date: str | None = None,
    era5_node_radius_km: float = 100.0,
    horizons_h: List[int] | None = None,
    threshold_mm: float = 1.0,
    era5_north_path: str | Path | None = None,
) -> CropOSDataset:
    """Build a CropOSDataset from on-disk or HuggingFace-cached parquet files.

    Args:
        era5_path:       Path to era5_thailand.parquet (southern/existing grid).
        nwp_path:        Path to nwp_features.parquet.
        station_order:   Ordered list of METAR station IDs.
        station_coords:  {station_id: (lat, lon)}.
        start_date:      Clip to timestamps >= start_date (ISO format, UTC).
        end_date:        Clip to timestamps <= end_date (ISO format, UTC).
        era5_node_radius_km: ERA5 spatial filtering radius (default 100 km).
        horizons_h:      Forecast horizons in hours.
        threshold_mm:    Rain/no-rain threshold.
        era5_north_path: Optional path to era5_north.parquet (northern grid top-up).
                         If provided, concatenated with era5_path before filtering.

    Returns:
        Constructed CropOSDataset ready for DataLoader.
    """
    logger.info(f"Loading ERA5 from {era5_path}...")
    era5_df = pd.read_parquet(era5_path)
    era5_df["timestamp"] = pd.to_datetime(era5_df["timestamp"], utc=True)
    if start_date:
        era5_df = era5_df[era5_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        era5_df = era5_df[era5_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

    # Merge northern top-up if available (adds grid points for Bangkok, Chiang Mai, etc.)
    if era5_north_path is not None and Path(era5_north_path).exists():
        logger.info(f"Loading ERA5 north top-up from {era5_north_path}...")
        north_df = pd.read_parquet(era5_north_path)
        north_df["timestamp"] = pd.to_datetime(north_df["timestamp"], utc=True)
        if start_date:
            north_df = north_df[north_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            north_df = north_df[north_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
        era5_df = pd.concat([era5_df, north_df], ignore_index=True)
        del north_df
        logger.info(
            f"ERA5 (south + north): {len(era5_df):,} rows, "
            f"{era5_df[['lat','lon']].drop_duplicates().__len__()} unique grid points"
        )
    else:
        logger.info(f"ERA5: {len(era5_df):,} rows, {era5_df['timestamp'].nunique():,} timestamps")

    logger.info(f"Loading NWP from {nwp_path}...")
    nwp_df = pd.read_parquet(nwp_path)
    nwp_df["timestamp"] = pd.to_datetime(nwp_df["timestamp"], utc=True)
    if start_date:
        nwp_df = nwp_df[nwp_df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        nwp_df = nwp_df[nwp_df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    logger.info(f"NWP: {len(nwp_df):,} rows, {nwp_df['station'].nunique()} stations")

    return CropOSDataset(
        era5_df=era5_df,
        nwp_df=nwp_df,
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
        return Path(hf_hub_download(  # nosec B615 — revision pinned to branch; SHA pinning is impractical for a live dataset
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
            revision="main",
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

    # Try the 22-var NWP features file first; fall back to the legacy baseline.
    try:
        nwp_path = _dl("nwp_features.parquet")
        logger.info("Using nwp_features.parquet (22-variable GFS set)")
    except Exception:
        nwp_path = _dl("nwp_baseline.parquet")
        logger.warning("nwp_features.parquet not found on HF — using legacy nwp_baseline.parquet")

    return load_dataset_from_parquets(
        era5_path=era5_path,
        nwp_path=nwp_path,
        station_order=station_order,
        station_coords=station_coords,
        start_date=start_date,
        end_date=end_date,
        era5_node_radius_km=era5_node_radius_km,
        horizons_h=horizons_h,
        threshold_mm=threshold_mm,
        era5_north_path=era5_north_path,
    )
