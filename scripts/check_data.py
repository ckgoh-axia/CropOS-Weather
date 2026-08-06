#!/usr/bin/env python3
"""Sanity-check the CropOS training parquets and identify data coverage gaps.

Usage
-----
    python scripts/check_data.py                    # reads from HF (needs HF_TOKEN)
    python scripts/check_data.py --local data/raw   # reads local parquet files
    python scripts/check_data.py --local data/raw --verbose

What it checks
--------------
  ERA5   : row count, grid point coverage, date range, variable completeness,
           rain fraction, and a timestamp gap scan (any hours missing?).
  METAR  : per-station row count, p01i coverage, rain fraction.
  NWP    : tries nwp_features.parquet (22 vars) then nwp_baseline.parquet (legacy).
           Per-station per-variable null rate, date range, expected vs actual rows.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HF_DATASET_REPO_NAME = "cropos-data"
EXPECTED_STATIONS = [
    "VTUU", "VTUD", "VTUK", "VTUB", "VTUN", "VTUL",
    "VTCC", "VTCP", "VTCN", "VTBS", "VTBD", "VTBP",
    "VTSS", "VTSP", "VTSH", "VTSG",
]
ERA5_EXPECTED_GRID_POINTS = 1980  # 60 lat × 33 lon at 0.25° spacing


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_hf(filename: str, repo_id: str, token: str) -> pd.DataFrame:
    # Stream directly from HF — no local disk cache, no ~/.cache/huggingface bloat
    url = (
        f"https://huggingface.co/datasets/{repo_id}"
        f"/resolve/main/{filename}"
    )
    import requests
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    import io
    return pd.read_parquet(io.BytesIO(resp.content))


def _load_local(filename: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / filename)


def _section(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print("═" * 64)


def _ok(msg: str)   -> None: print(f"  ✓  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠  {msg}")
def _err(msg: str)  -> None: print(f"  ✗  {msg}")


def _scan_timestamp_gaps(
    ts_series: pd.Series,
    freq: str = "1h",
    label: str = "data",
) -> int:
    """Print the first 10 timestamp gaps and return total gap count."""
    ts = pd.to_datetime(ts_series).dt.tz_localize(None)  # drop tz for comparison
    ts_sorted = ts.drop_duplicates().sort_values()
    full_range = pd.date_range(ts_sorted.iloc[0], ts_sorted.iloc[-1], freq=freq)
    missing = full_range.difference(ts_sorted)
    if len(missing) == 0:
        _ok(f"No {freq} gaps in {label} timestamp coverage")
    else:
        _warn(f"{len(missing):,} missing {freq} timestamps in {label}")
        for t in list(missing[:10]):
            print(f"       gap: {t}")
        if len(missing) > 10:
            print(f"       ... {len(missing) - 10} more gaps not shown")
    return len(missing)


# ── per-source checks ─────────────────────────────────────────────────────────

def check_era5(df: pd.DataFrame, verbose: bool = False) -> None:
    _section("ERA5 — atmospheric reanalysis grid")

    ts_col = "timestamp" if "timestamp" in df.columns else "time"
    df[ts_col] = pd.to_datetime(df[ts_col])

    n_pts = df.groupby(["lat", "lon"]).ngroups
    n_ts  = df[ts_col].nunique()

    print(f"  Rows            : {len(df):,}")
    print(f"  Unique grid pts : {n_pts:,}  (expected {ERA5_EXPECTED_GRID_POINTS})")
    print(f"  Unique timestamps: {n_ts:,}")
    print(f"  Date range      : {df[ts_col].min().date()} → {df[ts_col].max().date()}")
    print(f"  Lat range       : {df['lat'].min():.2f} → {df['lat'].max():.2f}")
    print(f"  Lon range       : {df['lon'].min():.2f} → {df['lon'].max():.2f}")

    if n_pts < ERA5_EXPECTED_GRID_POINTS:
        _warn(
            f"Only {n_pts} of {ERA5_EXPECTED_GRID_POINTS} expected grid points present. "
            f"ERA5 download may be incomplete."
        )
    else:
        _ok(f"All {n_pts} ERA5 grid points present")

    # Variable completeness
    skip = {"lat", "lon", ts_col}
    numeric_cols = [c for c in df.select_dtypes("number").columns if c not in skip]
    print(f"\n  {'Variable':<35}  {'Non-null%':>9}  {'Mean':>9}  {'Min':>9}  {'Max':>9}")
    print(f"  {'-'*35}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    all_complete = True
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        pct = 100.0 * s.notna().mean()
        if pct < 99.0:
            all_complete = False
        print(
            f"  {col:<35}  {pct:>8.1f}%  {s.mean():>9.3f}  {s.min():>9.3f}  {s.max():>9.3f}"
        )
    if all_complete:
        _ok("All ERA5 variable columns are >99% complete")

    # Rain fraction
    precip_cols = [c for c in numeric_cols if "precip" in c.lower()]
    if precip_cols:
        pcol = precip_cols[0]
        pct_rain = 100 * (df[pcol] >= 1.0).mean()
        msg = f"Rain fraction (≥1 mm, '{pcol}'): {pct_rain:.1f}%"
        if 5 < pct_rain < 50:
            _ok(msg)
        else:
            _warn(msg + "  ← expected 5–50% for Thailand")

    # Timestamp gap scan (only on the most-common lat/lon point, for speed)
    if verbose:
        sample_pt = df.groupby(["lat", "lon"]).size().idxmax()
        pt_df = df[(df["lat"] == sample_pt[0]) & (df["lon"] == sample_pt[1])]
        _scan_timestamp_gaps(pt_df[ts_col], label=f"ERA5 at ({sample_pt[0]}, {sample_pt[1]})")


def check_metar(df: pd.DataFrame, verbose: bool = False) -> None:
    _section("METAR — airport surface observations")

    ts_col = next(
        (c for c in df.columns if c in ("valid", "timestamp", "time")), None
    )
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])

    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {list(df.columns)}")
    if ts_col:
        print(f"  Date range : {df[ts_col].min().date()} → {df[ts_col].max().date()}")

    if "station" not in df.columns:
        _warn("No 'station' column — cannot do per-station analysis")
        return

    present_stations = sorted(df["station"].unique())
    missing_stations = [s for s in EXPECTED_STATIONS if s not in present_stations]
    print(f"\n  Stations present : {len(present_stations)}  "
          f"(expected {len(EXPECTED_STATIONS)})")
    if missing_stations:
        _warn(f"Missing stations: {missing_stations}")
    else:
        _ok("All 16 stations present")

    # Per-station coverage table
    print(f"\n  {'Station':<8}  {'Rows':>8}  {'p01i coverage':>14}  {'Rain ≥1mm':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*10}")
    for station in EXPECTED_STATIONS:
        grp = df[df["station"] == station]
        if len(grp) == 0:
            print(f"  {station:<8}  {'0':>8}  {'n/a':>14}  {'n/a':>10}  ← MISSING")
            continue
        if "p01i" in grp.columns:
            p01i_mm = pd.to_numeric(grp["p01i"], errors="coerce") * 25.4
            cov = f"{100 * p01i_mm.notna().mean():.0f}%"
            rain = f"{(p01i_mm >= 1.0).mean():.3f}"
        elif "precip_mm" in grp.columns:
            precip = pd.to_numeric(grp["precip_mm"], errors="coerce")
            cov = f"{100 * precip.notna().mean():.0f}%"
            rain = f"{(precip >= 1.0).mean():.3f}"
        else:
            cov = "n/a"
            rain = "n/a"
        print(f"  {station:<8}  {len(grp):>8,}  {cov:>14}  {rain:>10}")


def check_nwp(df: pd.DataFrame, filename: str = "nwp", verbose: bool = False) -> None:
    _section(f"NWP — GFS forecasts  ({filename})")

    ts_col = next(
        (c for c in df.columns if c in ("timestamp", "time", "valid")), None
    )
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])

    nwp_cols = [c for c in df.columns if c.startswith("nwp_")]
    print(f"  Rows         : {len(df):,}")
    print(f"  NWP columns  : {len(nwp_cols)}")
    if ts_col and len(df) > 0:
        print(f"  Date range   : {df[ts_col].min().date()} → {df[ts_col].max().date()}")
        print(f"  Unique ts    : {df[ts_col].nunique():,}")

    if len(df) == 0:
        _err("NWP file is EMPTY — all stations failed during download")
        print("     Re-run augment_data.py --nwp-only after checking Open-Meteo access")
        return

    if "station" in df.columns:
        present = sorted(df["station"].unique())
        missing = [s for s in EXPECTED_STATIONS if s not in present]
        print(f"\n  Stations : {len(present)}  (expected {len(EXPECTED_STATIONS)})")
        if missing:
            _warn(f"Missing stations in NWP: {missing}")
        else:
            _ok("All 16 stations present in NWP")

        # Expected rows: n_stations × n_timestamps
        n_ts = df[ts_col].nunique() if ts_col else 0
        expected_rows = len(present) * n_ts
        if abs(len(df) - expected_rows) / max(expected_rows, 1) > 0.01:
            _warn(
                f"Row count {len(df):,} ≠ expected {expected_rows:,} "
                f"({len(present)} stations × {n_ts} timestamps). "
                f"Some station/timestamp combinations may be missing."
            )
        else:
            _ok(f"Row count matches {len(present)} stations × {n_ts} timestamps")

    if not nwp_cols:
        _warn("No nwp_* columns found — check column naming")
        return

    # Per-variable null rate summary
    print(f"\n  {'NWP variable':<40}  {'Non-null%':>9}")
    print(f"  {'-'*40}  {'-'*9}")
    all_complete = True
    for col in nwp_cols:
        pct = 100.0 * df[col].notna().mean()
        if pct < 95.0:
            all_complete = False
        flag = "  ← ⚠" if pct < 95.0 else ""
        print(f"  {col:<40}  {pct:>8.1f}%{flag}")

    if all_complete:
        _ok("All NWP variables are ≥95% complete across all stations")

    if verbose and ts_col and "station" in df.columns:
        # Per-station timestamp gap scan
        for station in (present if "station" in df.columns else []):
            s_df = df[df["station"] == station]
            _scan_timestamp_gaps(s_df[ts_col], label=f"NWP {station}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Check CropOS training data coverage")
    parser.add_argument(
        "--local", metavar="DIR", default=None,
        help="Read from this local directory instead of HuggingFace"
    )
    parser.add_argument("--repo", default=None, help="HF repo id (overrides auto-detect)")
    parser.add_argument("--verbose", action="store_true", help="Enable timestamp gap scan")
    args = parser.parse_args()

    if args.local:
        data_dir = Path(args.local)
        print(f"Reading from local directory: {data_dir}/")

        def loader(name: str) -> pd.DataFrame:
            return _load_local(name, data_dir)

        def try_loader(*names: str) -> tuple[pd.DataFrame, str]:
            for name in names:
                path = data_dir / name
                if path.exists():
                    print(f"  Found: {name}")
                    return pd.read_parquet(path), name
            raise FileNotFoundError(
                f"None of {names} found in {data_dir}"
            )
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("Set HF_TOKEN or use --local <dir>")
        if args.repo:
            repo_id = args.repo
        else:
            from huggingface_hub import HfApi
            username = HfApi().whoami(token=token)["name"]
            repo_id = f"{username}/{HF_DATASET_REPO_NAME}"
        print(f"Reading from HuggingFace: hf://datasets/{repo_id}")

        def loader(name: str) -> pd.DataFrame:
            return _load_hf(name, repo_id, token)

        def try_loader(*names: str) -> tuple[pd.DataFrame, str]:
            for name in names:
                try:
                    df = _load_hf(name, repo_id, token)
                    print(f"  Found on HF: {name}")
                    return df, name
                except Exception:
                    continue
            raise FileNotFoundError(f"None of {names} found on HF repo {repo_id}")

    # ERA5
    era5_df = loader("era5_thailand.parquet")
    check_era5(era5_df, verbose=args.verbose)

    # METAR
    metar_df = loader("metar_thai.parquet")
    check_metar(metar_df, verbose=args.verbose)

    # NWP — try 22-var file first, fall back to legacy
    nwp_df, nwp_file = try_loader("nwp_features.parquet", "nwp_baseline.parquet")
    check_nwp(nwp_df, filename=nwp_file, verbose=args.verbose)

    # Summary
    _section("Summary")
    total = len(era5_df) + len(metar_df) + len(nwp_df)
    print(f"  ERA5   : {len(era5_df):>12,} rows")
    print(f"  METAR  : {len(metar_df):>12,} rows")
    print(f"  NWP    : {len(nwp_df):>12,} rows  ({nwp_file})")
    print(f"  Total  : {total:>12,} rows")
    print()


if __name__ == "__main__":
    main()
