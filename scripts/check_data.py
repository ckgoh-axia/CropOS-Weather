#!/usr/bin/env python3
"""Sanity-check the three CropOS parquet files downloaded from HuggingFace Datasets.

Usage:
    python scripts/check_data.py                # reads from HF (needs HF_TOKEN)
    python scripts/check_data.py --local data/raw  # reads local parquet files
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_hf(filename: str, repo_id: str, token: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", token=token
    )
    return pd.read_parquet(path)


def _load_local(filename: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / filename)


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def check_era5(df: pd.DataFrame) -> None:
    section("ERA5 — atmospheric grid")
    ts_col = "timestamp" if "timestamp" in df.columns else "time"
    df[ts_col] = pd.to_datetime(df[ts_col])

    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"  Date range    : {df[ts_col].min().date()} → {df[ts_col].max().date()}")
    print(f"  Unique points : {df.groupby(['lat','lon']).ngroups:,}")
    print(f"  Lat range     : {df['lat'].min():.2f} → {df['lat'].max():.2f}")
    print(f"  Lon range     : {df['lon'].min():.2f} → {df['lon'].max():.2f}")

    # Check each ERA5 variable
    numeric = df.select_dtypes(include="number").columns.drop(["lat", "lon"], errors="ignore")
    print(f"\n  {'Variable':<30}  {'Non-null%':>9}  {'Mean':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'-'*30}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*10}")
    for col in numeric:
        s = df[col].dropna()
        pct = 100 * len(s) / len(df)
        print(f"  {col:<30}  {pct:>8.1f}%  {s.mean():>10.3f}  {s.min():>10.3f}  {s.max():>10.3f}")

    # Precipitation sanity: should have some rain in Thailand
    precip_cols = [c for c in numeric if "precip" in c.lower() or "precipitation" in c.lower()]
    if precip_cols:
        pcol = precip_cols[0]
        rain_frac = (df[pcol] >= 1.0).mean()
        print(f"\n  Rain fraction (≥1 mm, '{pcol}'): {rain_frac:.3f}  "
              f"({'✓ looks plausible' if 0.05 < rain_frac < 0.5 else '⚠ check units/values'})")


def check_metar(df: pd.DataFrame) -> None:
    section("METAR — airport surface observations")
    ts_col = next((c for c in df.columns if c in ("valid", "timestamp", "time")), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])

    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {list(df.columns)}")
    if ts_col:
        print(f"  Date range : {df[ts_col].min().date()} → {df[ts_col].max().date()}")

    # Per-station coverage
    if "station" in df.columns:
        print(f"\n  Stations  : {df['station'].nunique()}")
        rows = []
        for station, grp in df.groupby("station"):
            p01i = (
                pd.to_numeric(grp["p01i"], errors="coerce") * 25.4
                if "p01i" in grp.columns
                else pd.Series(dtype=float)
            )
            rows.append({
                "station": station,
                "rows": len(grp),
                "p01i_coverage": f"{100 * p01i.notna().mean():.0f}%" if len(p01i) else "n/a",
                "rain_fraction": f"{(p01i >= 1.0).mean():.3f}" if len(p01i) else "n/a",
            })
        print(f"\n  {'Station':<8}  {'Rows':>8}  {'p01i coverage':>14}  {'Rain frac (≥1mm)':>17}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*17}")
        for row in rows:
            print(
                f"  {row['station']:<8}  {row['rows']:>8,}"
                f"  {row['p01i_coverage']:>14}  {row['rain_fraction']:>17}"
            )

    if "p01i" in df.columns:
        p01i_mm = pd.to_numeric(df["p01i"], errors="coerce") * 25.4
        overall_rain = (p01i_mm >= 1.0).mean()
        print(f"\n  Overall rain fraction (p01i ≥ 1mm): {overall_rain:.3f}")
        print(f"  Max hourly precip (mm)             : {p01i_mm.max():.1f}")
        print(f"  p01i null fraction                 : {df['p01i'].isna().mean():.3f}")


def check_nwp(df: pd.DataFrame) -> None:
    section("NWP baseline — GFS via Open-Meteo (16 stations)")
    ts_col = next((c for c in df.columns if c in ("timestamp", "time", "valid")), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])

    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {list(df.columns)}")
    if ts_col and len(df) > 0:
        print(f"  Date range : {df[ts_col].min().date()} → {df[ts_col].max().date()}")
    if "station" in df.columns:
        print(f"  Stations  : {df['station'].nunique()}  {sorted(df['station'].unique())}")

    # NWP precip column
    nwp_cols = [c for c in df.columns if "precip" in c.lower() or "nwp" in c.lower()]
    if nwp_cols and len(df) > 0:
        col = nwp_cols[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        rain_frac = (s >= 1.0).mean()
        print(f"\n  NWP precip col '{col}': mean={s.mean():.3f} mm, max={s.max():.1f} mm, "
              f"rain_frac={rain_frac:.3f}")

    if len(df) == 0:
        print("\n  ⚠  NWP file is EMPTY — all stations failed during download.")
        print("     Re-run download_data.py after confirming Open-Meteo historical API access.")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check CropOS training data")
    parser.add_argument("--local", metavar="DIR", default=None,
                        help="Read from local directory instead of HuggingFace")
    parser.add_argument("--repo", default=None,
                        help="HF repo id (default: auto-detect from token)")
    args = parser.parse_args()

    if args.local:
        data_dir = Path(args.local)
        def loader(name):
            return _load_local(name, data_dir)
        print(f"Reading from local directory: {data_dir}/")
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("Set HF_TOKEN or use --local <dir>")
        if args.repo:
            repo_id = args.repo
        else:
            from huggingface_hub import HfApi
            username = HfApi().whoami(token=token)["name"]
            repo_id = f"{username}/cropos-data"
        print(f"Reading from HuggingFace: hf://datasets/{repo_id}")
        def loader(name):
            return _load_hf(name, repo_id, token)

    era5_df = loader("era5_thailand.parquet")
    metar_df = loader("metar_thai.parquet")
    nwp_df = loader("nwp_baseline.parquet")

    check_era5(era5_df)
    check_metar(metar_df)
    check_nwp(nwp_df)

    section("Summary")
    total = len(era5_df) + len(metar_df) + len(nwp_df)
    print(f"  ERA5   : {len(era5_df):>10,} rows")
    print(f"  METAR  : {len(metar_df):>10,} rows")
    print(f"  NWP    : {len(nwp_df):>10,} rows")
    print(f"  Total  : {total:>10,} rows")
    print()


if __name__ == "__main__":
    main()
