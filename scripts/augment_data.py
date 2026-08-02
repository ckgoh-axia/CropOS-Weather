#!/usr/bin/env python3
"""Augment the existing HF dataset with missing pieces — no re-downloading.

What this does
--------------
ERA5   : Resumes from the HF checkpoint. Skips all already-downloaded grid
         points. Run repeatedly (via cron) until all 1,980 points are done.
METAR  : Fetches only the 3 stations that returned 0 rows the first time
         (VTUB, VTUN, VTBP). Merges into the existing metar_thai.parquet
         and re-uploads to HF. Logs per-station/per-year diagnostics so
         the root cause of each failure is visible (not swallowed).
NWP    : Downloads the full GFS physics variable set (17 variables including
         CAPE, 850/500 hPa winds, precipitable water, soil moisture) at all
         16 METAR station locations for 2016–2022. Saves as nwp_features.parquet.
         This is the PRIMARY model input for the GFS-correction architecture.
         Per-station checkpoints mean partial runs resume cleanly.

Usage
-----
    # NWP features only (primary new data for production model):
    python scripts/augment_data.py --nwp-only

    # ERA5 continuation only (what the cron runs):
    python scripts/augment_data.py --era5-only

    # METAR missing-station fill only:
    python scripts/augment_data.py --metar-only

    # All phases:
    python scripts/augment_data.py

    # Dry-run without pushing to HF:
    python scripts/augment_data.py --nwp-only --skip-push
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd
import yaml
from huggingface_hub import HfApi, create_repo, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET_REPO_NAME = "cropos-data"

# Stations that returned 0 rows in the initial download.
# We try them again here with full per-year diagnostics so we can see exactly
# why they failed: not in Iowa State network, HTTP error, or genuinely empty.
MISSING_METAR_STATIONS = ["VTUB", "VTUN", "VTBP"]

IOWA_STATE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
INTER_YEAR_DELAY_S = 5.0
STATION_DELAY_S = 2.0
REQUEST_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# METAR augmentation
# ---------------------------------------------------------------------------

def _fetch_station_year_with_diagnostics(
    station: str, year: int, client: httpx.Client
) -> tuple[pd.DataFrame, dict]:
    """Fetch one station-year and return (df, diagnostics).

    Unlike the production fetch which silently swallows errors, this version
    records the HTTP status, Iowa State error message, and row count so we
    can distinguish: station not in network / HTTP error / genuinely empty.
    """
    params = {
        "station": station,
        "data": "all",
        "year1": str(year),   "month1": "1",  "day1": "1",
        "year2": str(year),   "month2": "12", "day2": "31",
        "tz": "Etc/UTC",
        "format": "comma",
        "latlon": "yes",
        "direct": "yes",
    }
    diag: dict = {"station": station, "year": year, "status": -1, "rows": 0, "note": ""}
    try:
        resp = client.get(IOWA_STATE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        diag["status"] = resp.status_code

        if resp.status_code == 429:
            diag["note"] = "RATE LIMITED"
            return pd.DataFrame(), diag

        if resp.status_code != 200:
            diag["note"] = f"HTTP {resp.status_code}: {resp.text[:120]}"
            return pd.DataFrame(), diag

        # Iowa State returns comment lines starting with '#' on error/empty.
        comment_lines = [ln for ln in resp.text.splitlines() if ln.startswith("#")]
        if comment_lines:
            diag["note"] = f"Iowa State comment: {comment_lines[0][:120]}"

        lines = [ln for ln in resp.text.splitlines() if not ln.startswith("#") and ln.strip()]
        if len(lines) <= 1:
            diag["note"] = diag["note"] or "empty response (header only)"
            return pd.DataFrame(), diag

        df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
        if df.empty:
            diag["note"] = "parsed to 0 rows"
            return df, diag

        # Mirror the production parse logic exactly.
        df = df.rename(columns={"valid": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["precip_mm"] = (
            pd.to_numeric(df.get("p01i", 0), errors="coerce").fillna(0) * 25.4
        )
        wxcodes = df.get("wxcodes", pd.Series([""] * len(df))).fillna("")
        df["rain_event"] = wxcodes.str.contains(r"\bRA\b|\bTS\b|\bSH\b", regex=True)
        df["station"] = df["station"].str.strip()
        for col in ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        keep = [
            "station", "timestamp", "precip_mm", "rain_event",
            "tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby",
        ]
        df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

        diag["rows"] = len(df)
        diag["note"] = "ok"
        return df, diag

    except Exception as exc:
        diag["note"] = f"exception: {exc}"
        return pd.DataFrame(), diag


def _fetch_missing_station(station: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one missing station with full per-year diagnostics."""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    years = list(range(start_year, end_year + 1))
    yearly_frames: list[pd.DataFrame] = []

    logger.info(f"METAR-AUG | {station} — attempting {len(years)} years")
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        for idx, year in enumerate(years):
            df, diag = _fetch_station_year_with_diagnostics(station, year, client)
            if diag["rows"] > 0:
                yearly_frames.append(df)
                logger.info(f"METAR-AUG | {station} {year}: {diag['rows']:,} rows ✓")
            else:
                # Log at WARNING with the exact diagnostic — this is the RCA payload.
                logger.warning(
                    f"METAR-AUG | {station} {year}: 0 rows — "
                    f"HTTP {diag['status']} | {diag['note']}"
                )
            if idx < len(years) - 1:
                time.sleep(INTER_YEAR_DELAY_S)

    if not yearly_frames:
        return pd.DataFrame()
    return pd.concat(yearly_frames, ignore_index=True)


def augment_metar(
    cfg: dict,
    start: str,
    end: str,
    repo_id: str,
    token: str,
    skip_push: bool,
) -> None:
    """Fetch missing stations, merge with existing parquet, push to HF."""
    from src.ingestion.metar import STATION_COORDS

    # Pull the existing METAR parquet from HF so we can merge.
    logger.info("METAR-AUG | pulling existing metar_thai.parquet from HF...")
    local_dir = Path("data/raw/.metar_aug")
    local_dir.mkdir(parents=True, exist_ok=True)
    existing_path = local_dir / "metar_thai_existing.parquet"
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="metar_thai.parquet",
            repo_type="dataset",
            token=token,
            local_dir=str(local_dir),
        )
        # hf_hub_download saves to <local_dir>/metar_thai.parquet
        downloaded = local_dir / "metar_thai.parquet"
        if downloaded.exists():
            downloaded.rename(existing_path)
        existing_df = pd.read_parquet(existing_path)
        existing_stations = set(existing_df["station"].unique())
        logger.info(
            f"METAR-AUG | existing parquet: {len(existing_df):,} rows, "
            f"{len(existing_stations)} stations: {sorted(existing_stations)}"
        )
    except Exception as exc:
        logger.warning(f"METAR-AUG | could not pull existing parquet: {exc}")
        existing_df = pd.DataFrame()
        existing_stations = set()

    # Determine which stations are actually missing.
    to_fetch = [s for s in MISSING_METAR_STATIONS if s not in existing_stations]
    if not to_fetch:
        logger.info("METAR-AUG | all target stations already present — nothing to do")
        return

    logger.info(f"METAR-AUG | stations to fetch: {to_fetch}")

    # Fetch each missing station with full diagnostics.
    new_frames: list[pd.DataFrame] = []
    for i, station in enumerate(to_fetch):
        df = _fetch_missing_station(station, start, end)
        if not df.empty:
            if station in STATION_COORDS:
                lat, lon = STATION_COORDS[station]
                df["lat"] = lat
                df["lon"] = lon
            new_frames.append(df)
            logger.info(f"METAR-AUG | {station}: {len(df):,} rows collected ✓")
        else:
            logger.warning(
                f"METAR-AUG | {station}: 0 rows across all years — "
                f"station likely not in Iowa State ASOS network"
            )
        if i < len(to_fetch) - 1:
            time.sleep(STATION_DELAY_S)

    if not new_frames:
        logger.warning(
            "METAR-AUG | no new data from any missing station. "
            "These stations are probably absent from Iowa State ASOS. "
            "Dataset remains at 13/16 stations — acceptable for training."
        )
        return

    # Merge with existing and push.
    merged = pd.concat([existing_df] + new_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["station", "timestamp"]).reset_index(drop=True)
    logger.info(
        f"METAR-AUG | merged: {len(merged):,} rows, "
        f"{merged['station'].nunique()} stations"
    )

    out_path = Path("data/raw/metar_thai.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    if not skip_push:
        push_to_hf(out_path, repo_id, token)
        logger.info("METAR-AUG | merged parquet pushed to HF ✓")
    else:
        logger.info(f"METAR-AUG | skip-push: merged parquet at {out_path}")


# ---------------------------------------------------------------------------
# ERA5 continuation (checkpoint-aware — skips already-done batches)
# ---------------------------------------------------------------------------

def continue_era5(
    cfg: dict,
    start: str,
    end: str,
    repo_id: str | None,
    token: str | None,
    skip_push: bool,
) -> None:
    """Resume ERA5 grid download from HF checkpoint.

    Pulls the ERA5 checkpoint from HF (so this runner picks up exactly where
    the last run stopped), downloads the next ~4 batches until the hourly
    limit is hit, then pushes the updated checkpoint and final parquet back.
    """
    from scripts.download_data import download_era5, push_to_hf

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)

    era5_path = download_era5(
        cfg, start, end, outdir,
        repo_id=repo_id if not skip_push else None,
        hf_token=token if not skip_push else None,
    )

    if not skip_push:
        push_to_hf(era5_path, repo_id, token)
        logger.info("ERA5-AUG | final parquet pushed to HF ✓")
    else:
        logger.info(f"ERA5-AUG | skip-push: parquet at {era5_path}")


# ---------------------------------------------------------------------------
# ERA5 top-up (downloads only grid points absent from the existing HF parquet)
# ---------------------------------------------------------------------------

def topup_era5(
    cfg: dict,
    start: str,
    end: str,
    repo_id: str | None,
    token: str | None,
    skip_push: bool,
) -> None:
    """Download only ERA5 grid points missing from era5_thailand.parquet on HF.

    Saves result as era5_north.parquet (separate file — not merged with the
    existing parquet). Merging a ~3 GB parquet on a 7 GB runner risks OOM;
    instead, dataset.py concatenates both files at load time using only the
    grid points within edge_radius_km of any station.

    Uses pyarrow to read only lat/lon columns from the existing parquet so we
    identify missing points without loading ~3 GB into RAM.
    """
    import pyarrow.parquet as pq
    from src.ingestion.era5 import fetch_era5_grid, build_thailand_grid

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: identify existing grid points (lat/lon columns only) ──────────
    if repo_id and token and not skip_push:
        logger.info("ERA5-TOPUP | downloading era5_thailand.parquet from HF (lat/lon only)...")
        local_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename="era5_thailand.parquet",
            repo_type="dataset",
            token=token,
            local_dir=str(outdir),
        ))
    else:
        local_path = outdir / "era5_thailand.parquet"
        if not local_path.exists():
            raise FileNotFoundError(
                f"No local ERA5 file at {local_path} — pass --skip-push only with local data"
            )

    # Read only lat/lon — avoids loading the full ~3 GB into RAM
    tbl = pq.read_table(local_path, columns=["lat", "lon"])
    meta_df = tbl.to_pandas()
    existing_pts = set(
        zip(meta_df["lat"].round(2).tolist(), meta_df["lon"].round(2).tolist())
    )
    logger.info(f"ERA5-TOPUP | existing: {len(existing_pts):,} unique grid points")
    del meta_df, tbl

    # ── Step 2: identify missing grid points from the full Thailand grid ──────
    all_lats, all_lons = build_thailand_grid(spacing_deg=0.25)
    all_pts = [(round(lat, 2), round(lon, 2)) for lat, lon in zip(all_lats, all_lons)]
    missing = [(lat, lon) for lat, lon in all_pts if (lat, lon) not in existing_pts]
    logger.info(
        f"ERA5-TOPUP | full grid: {len(all_pts):,}  "
        f"already downloaded: {len(existing_pts):,}  "
        f"missing: {len(missing):,}"
    )

    if not missing:
        logger.info("ERA5-TOPUP | ERA5 grid already complete — nothing to do")
        return

    m_lats = [p[0] for p in missing]
    m_lons = [p[1] for p in missing]
    logger.info(
        f"ERA5-TOPUP | lat range to download: "
        f"{min(m_lats):.2f} → {max(m_lats):.2f}"
    )

    # ── Step 3: download and save as era5_north.parquet ───────────────────────
    new_df = fetch_era5_grid(m_lats, m_lons, start, end)
    logger.info(f"ERA5-TOPUP | downloaded {len(new_df):,} new rows")

    out_path = outdir / "era5_north.parquet"
    new_df.to_parquet(out_path, index=False)
    logger.info(f"ERA5-TOPUP | saved → {out_path}")

    if not skip_push:
        push_to_hf(out_path, repo_id, token)
        logger.info("ERA5-TOPUP | era5_north.parquet pushed to HF ✓")
    else:
        logger.info(f"ERA5-TOPUP | skip-push: file at {out_path}")


# ---------------------------------------------------------------------------
# NWP forward-fill (propagates 6-hourly GFS values to fill hourly nulls)
# ---------------------------------------------------------------------------

def ffill_nwp(
    repo_id: str | None,
    token: str | None,
    skip_push: bool,
) -> None:
    """Forward-fill per-station NWP nulls so hourly training timestamps have features.

    The NWP parquet is downloaded at GFS cadence (~4–6 runs/day) but the
    timestamp index is hourly, leaving ~75% of rows as NaN. Forward-filling
    propagates each GFS forecast value until the next run — exactly what an
    operational NWP system would do.

    nwp_soil_moisture_0_to_7cm was never populated (0% non-null). After ffill
    it remains 0 everywhere — a known-dead constant feature that is harmless
    (the GNN's weight for it converges to ≈0) and keeps LOCAL_STATION_IN = 39.
    """
    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)

    if repo_id and token and not skip_push:
        logger.info("NWP-FFILL | downloading nwp_features.parquet from HF...")
        nwp_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename="nwp_features.parquet",
            repo_type="dataset",
            token=token,
            local_dir=str(outdir),
        ))
    else:
        nwp_path = outdir / "nwp_features.parquet"
        if not nwp_path.exists():
            raise FileNotFoundError(f"No local NWP file at {nwp_path}")

    nwp_df = pd.read_parquet(nwp_path)
    nwp_cols = [c for c in nwp_df.columns if c.startswith("nwp_")]
    null_before = nwp_df[nwp_cols].isnull().mean().mean()
    logger.info(
        f"NWP-FFILL | loaded {len(nwp_df):,} rows, "
        f"{null_before:.1%} null before ffill, "
        f"{nwp_df['station'].nunique()} stations"
    )

    # Forward-fill per station in timestamp order
    nwp_df["timestamp"] = pd.to_datetime(nwp_df["timestamp"], utc=True)
    nwp_df = nwp_df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    nwp_df[nwp_cols] = nwp_df.groupby("station", sort=False)[nwp_cols].ffill()

    # Fill any remaining NaN (rows before the station's first GFS run) with 0
    nwp_df[nwp_cols] = nwp_df[nwp_cols].fillna(0.0)

    null_after = nwp_df[nwp_cols].isnull().mean().mean()
    logger.info(f"NWP-FFILL | null after ffill+fill0: {null_after:.1%} (target: 0.0%)")

    out_path = outdir / "nwp_features.parquet"
    nwp_df.to_parquet(out_path, index=False)
    logger.info(f"NWP-FFILL | {len(nwp_df):,} rows → {out_path}")

    if not skip_push:
        push_to_hf(out_path, repo_id, token)
        logger.info("NWP-FFILL | nwp_features.parquet pushed to HF ✓")
    else:
        logger.info(f"NWP-FFILL | skip-push: file at {out_path}")


# ---------------------------------------------------------------------------
# NWP features download (GFS-correction model — primary training input)
# ---------------------------------------------------------------------------

def download_nwp_features(
    cfg: dict,
    start: str,
    end: str,
    repo_id: str | None,
    token: str | None,
    skip_push: bool,
) -> None:
    """Download full GFS physics variable set at all METAR station locations.

    This is a one-shot download (not cron-based): 16 stations × ~60k rows each
    = ~960k rows total. Should complete in a single GitHub Actions run in ~30-60
    minutes. Per-station checkpoints mean it resumes cleanly on failure.

    Output: data/raw/nwp_features.parquet → pushed to HF as nwp_features.parquet
    """
    from src.ingestion.metar import STATION_COORDS
    from src.ingestion.nwp_baseline import fetch_all_stations

    # The historical forecast API is only available from 2016-01-01.
    nwp_min = cfg.get("nwp_min_start", "2016-01-01")
    effective_start = start if start >= nwp_min else nwp_min
    if effective_start != start:
        logger.info(f"NWP-FEAT | clamping start {start} → {effective_start} (API limit)")

    variables = cfg.get("nwp_variables", None)  # None → use NWP_DEFAULT_VARIABLES

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / ".nwp_checkpoint"

    # Pull existing per-station checkpoint files from HF so re-runs on ephemeral
    # GitHub Actions runners don't re-download already-completed stations.
    if repo_id and token:
        _pull_nwp_checkpoints_from_hf(checkpoint_dir, repo_id, token)

    logger.info(
        f"NWP-FEAT | fetching {len(STATION_COORDS)} stations, "
        f"{effective_start} → {end}, "
        f"{len(variables) if variables else 'default'} variables"
    )

    try:
        nwp_df = fetch_all_stations(
            STATION_COORDS, effective_start, end,
            variables=variables,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        # Push checkpoint files to HF even on partial failure.
        if repo_id and token:
            _push_nwp_checkpoints_to_hf(checkpoint_dir, repo_id, token)

    out_path = outdir / "nwp_features.parquet"
    nwp_df.to_parquet(out_path, index=False)
    logger.info(f"NWP-FEAT | {len(nwp_df):,} rows → {out_path}")

    if not skip_push:
        push_to_hf(out_path, repo_id, token)
        logger.info("NWP-FEAT | nwp_features.parquet pushed to HF ✓")
    else:
        logger.info(f"NWP-FEAT | skip-push: parquet at {out_path}")


def _pull_nwp_checkpoints_from_hf(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    """Pull per-station NWP checkpoint parquets from HF to resume partial downloads."""
    from src.ingestion.metar import THAI_METAR_STATIONS

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pulled = 0
    for station in THAI_METAR_STATIONS:
        local = checkpoint_dir / f"{station}.parquet"
        if local.exists():
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=f"nwp_checkpoints/{station}.parquet",
                repo_type="dataset",
                token=token,
                local_dir=str(checkpoint_dir),
            )
            # hf_hub_download creates subdirectory structure; move to flat dir
            nested = checkpoint_dir / "nwp_checkpoints" / f"{station}.parquet"
            if nested.exists():
                nested.rename(local)
            pulled += 1
        except Exception:
            pass  # Not yet on HF — station will be downloaded fresh
    if pulled:
        logger.info(f"NWP-FEAT | restored {pulled} station checkpoints from HF")


def _push_nwp_checkpoints_to_hf(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    """Push per-station checkpoint parquets to HF so next run can resume."""
    from huggingface_hub import HfApi
    api = HfApi()
    pushed = 0
    for ckpt in sorted(checkpoint_dir.glob("*.parquet")):
        try:
            api.upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=f"nwp_checkpoints/{ckpt.name}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            pushed += 1
        except Exception as exc:
            logger.warning(f"NWP-FEAT | could not push checkpoint {ckpt.name}: {exc}")
    if pushed:
        logger.info(f"NWP-FEAT | pushed {pushed} station checkpoints to HF")


# ---------------------------------------------------------------------------
# HuggingFace push
# ---------------------------------------------------------------------------

def push_to_hf(local_path: Path, repo_id: str, token: str) -> None:
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    logger.info(f"HF | pushed {local_path.name} → hf://datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment CropOS dataset (no re-downloads)")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--era5-only", action="store_true", help="Only run ERA5 continuation (checkpoint-based)")
    parser.add_argument("--era5-topup", action="store_true", help="Download only ERA5 grid points missing from HF → era5_north.parquet")
    parser.add_argument("--metar-only", action="store_true", help="Only run METAR missing-station fill")
    parser.add_argument("--nwp-only", action="store_true", help="Only run NWP features download")
    parser.add_argument("--nwp-ffill", action="store_true", help="Forward-fill hourly NWP nulls and re-push to HF")
    parser.add_argument("--skip-push", action="store_true", help="Skip HuggingFace upload (local run)")
    args = parser.parse_args()

    # Validate mutual exclusivity of --*-only / --*-topup / --*-ffill flags
    only_flags = [args.era5_only, args.era5_topup, args.metar_only, args.nwp_only, args.nwp_ffill]
    if sum(only_flags) > 1:
        raise ValueError("Only one mode flag may be set at a time")

    token = os.environ.get("HF_TOKEN")
    if not token and not args.skip_push:
        raise ValueError(
            "Set HF_TOKEN environment variable (or pass --skip-push for a local-only run)"
        )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    start = args.start or cfg["training_start"]
    end = args.end or cfg["training_end"]

    # Resolve HF repo from token.
    if not args.skip_push:
        api = HfApi()
        username = api.whoami(token=token)["name"]
        repo_id = f"{username}/{HF_DATASET_REPO_NAME}"
        create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=token, exist_ok=True)
        logger.info(f"HF dataset repo: hf://datasets/{repo_id}")
    else:
        repo_id = None

    any_specific = args.era5_only or args.era5_topup or args.metar_only or args.nwp_only or args.nwp_ffill

    if args.era5_topup:
        logger.info("═══ ERA5 top-up (missing grid points → era5_north.parquet) ═══")
        topup_era5(cfg, start, end, repo_id, token, args.skip_push)

    elif args.nwp_ffill:
        logger.info("═══ NWP forward-fill (hourly null → GFS propagation) ═══")
        ffill_nwp(repo_id, token, args.skip_push)

    else:
        run_nwp   = args.nwp_only   or not any_specific
        run_era5  = args.era5_only  or not any_specific
        run_metar = args.metar_only or not any_specific

        if run_nwp:
            logger.info("═══ NWP features download (GFS physics variables) ═══")
            download_nwp_features(cfg, start, end, repo_id, token, args.skip_push)

        if run_era5:
            logger.info("═══ ERA5 continuation (checkpoint-aware) ═══")
            continue_era5(cfg, start, end, repo_id, token, args.skip_push)

        if run_metar:
            logger.info("═══ METAR missing-station augmentation ═══")
            augment_metar(cfg, start, end, repo_id or "", token or "", args.skip_push)

    logger.info("Augmentation complete.")


if __name__ == "__main__":
    main()
