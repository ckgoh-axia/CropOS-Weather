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
    parser.add_argument("--era5-only", action="store_true", help="Only run ERA5 continuation")
    parser.add_argument("--metar-only", action="store_true", help="Only run METAR missing-station fill")
    parser.add_argument("--nwp-only", action="store_true", help="Only run NWP features download")
    parser.add_argument("--skip-push", action="store_true", help="Skip HuggingFace upload (local run)")
    args = parser.parse_args()

    # Validate mutual exclusivity of --*-only flags
    only_flags = [args.era5_only, args.metar_only, args.nwp_only]
    if sum(only_flags) > 1:
        raise ValueError("Only one --*-only flag may be set at a time")

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

    run_nwp = args.nwp_only or not (args.era5_only or args.metar_only)
    run_era5 = args.era5_only or not (args.nwp_only or args.metar_only)
    run_metar = args.metar_only or not (args.nwp_only or args.era5_only)

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
