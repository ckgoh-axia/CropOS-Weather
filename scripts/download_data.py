#!/usr/bin/env python3
"""Download all CropOS training data and push to HuggingFace Datasets.

Sources (all free, no special accounts):
  - ERA5-Land via Open-Meteo: atmospheric features + precipitation labels
  - METAR via Iowa State ASOS: surface observations at 16 Thai airports
  - NWP baseline via Open-Meteo: GFS forecasts for Brier Skill Score comparison

ERA5 is rate-limited to ~5 requests/hour on Open-Meteo's free archive API.
The checkpoint is persisted to HuggingFace after every batch so GitHub Actions
re-runs resume exactly where the previous run stopped rather than restarting.
See .github/workflows/download_era5_cron.yml for the scheduled auto-resume.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET_REPO_NAME = "cropos-data"


# ---------------------------------------------------------------------------
# ERA5 checkpoint helpers
# ---------------------------------------------------------------------------

def _pull_era5_checkpoint_from_hf(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    """Pull ERA5 checkpoint from HuggingFace if available.

    GitHub Actions runners are ephemeral — without this, every run restarts
    ERA5 from batch 1 and hits the hourly limit before making real progress.
    With this, each run resumes from the last saved checkpoint.
    """
    from huggingface_hub import hf_hub_download

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "era5_checkpoint.parquet"

    if checkpoint_path.exists():
        logger.info("ERA5 | local checkpoint found — skipping HF pull")
        return

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="era5_checkpoint.parquet",
            repo_type="dataset",
            token=token,
            local_dir=str(checkpoint_dir),
        )
        size_mb = checkpoint_path.stat().st_size / 1024 / 1024
        logger.info(f"ERA5 | checkpoint restored from HF ({size_mb:.1f} MB) — resuming")
    except Exception as exc:
        logger.info(f"ERA5 | no HF checkpoint found (starting fresh): {exc}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_era5(
    cfg: dict,
    start: str,
    end: str,
    outdir: Path,
    repo_id: str | None = None,
    hf_token: str | None = None,
) -> Path:
    from src.ingestion.era5 import build_thailand_grid, fetch_era5_grid

    checkpoint_dir = outdir / ".era5_checkpoint"

    # Restore checkpoint from HF so this run resumes from where the last one stopped.
    if repo_id and hf_token:
        _pull_era5_checkpoint_from_hf(checkpoint_dir, repo_id, hf_token)

    logger.info("ERA5 | starting download (0.25° grid over Thailand)...")
    lat_pts, lon_pts = build_thailand_grid(spacing_deg=0.25)

    try:
        era5_df = fetch_era5_grid(lat_pts, lon_pts, start, end, checkpoint_dir=checkpoint_dir)
    finally:
        # Always push checkpoint to HF — even on partial runs (hourly rate limit hit).
        # This is what makes the next retrigger resume rather than restart.
        if repo_id and hf_token:
            checkpoint_path = checkpoint_dir / "era5_checkpoint.parquet"
            if checkpoint_path.exists():
                push_to_hf(checkpoint_path, repo_id, hf_token)
                logger.info("ERA5 | checkpoint pushed to HF — next run will resume here")

    path = outdir / "era5_thailand.parquet"
    era5_df.to_parquet(path, index=False)
    logger.info(f"ERA5 | {len(era5_df):,} rows → {path}")
    return path


def download_metar(cfg: dict, start: str, end: str, outdir: Path) -> Path:
    from src.ingestion.metar import fetch_all_thai_stations

    logger.info(f"METAR | starting download {start} → {end}")
    metar_df = fetch_all_thai_stations(start, end)
    path = outdir / "metar_thai.parquet"
    metar_df.to_parquet(path, index=False)
    logger.info(f"METAR | {len(metar_df):,} rows → {path}")
    return path


def download_nwp(cfg: dict, start: str, end: str, outdir: Path) -> Path:
    import pandas as pd

    from src.ingestion.metar import STATION_COORDS
    from src.ingestion.nwp_baseline import fetch_nwp_at_point

    # Open-Meteo historical forecast API only available from 2016-01-01
    NWP_MIN_START = "2016-01-01"
    effective_start = start if start >= NWP_MIN_START else NWP_MIN_START
    if effective_start != start:
        logger.info(f"NWP | clamping start {start} → {effective_start} (API limit)")

    logger.info("NWP | starting download (GFS via Open-Meteo, 16 stations)...")
    frames = []
    for station, (lat, lon) in STATION_COORDS.items():
        try:
            df = fetch_nwp_at_point(lat, lon, effective_start, end)
            df["station"] = station
            frames.append(df)
            logger.info(f"NWP | {station} done")
            time.sleep(5)  # stay under Open-Meteo free tier rate limit
        except Exception as exc:
            logger.warning(f"NWP | {station} failed: {exc}")
            time.sleep(10)  # back off longer on failure

    out_path = outdir / "nwp_baseline.parquet"
    if not frames:
        logger.warning("NWP | all stations failed — saving empty baseline file")
        pd.DataFrame(columns=["station", "timestamp", "nwp_precip_mm"]).to_parquet(
            out_path, index=False
        )
    else:
        nwp_df = pd.concat(frames, ignore_index=True)
        nwp_df.to_parquet(out_path, index=False)
        logger.info(f"NWP | {len(nwp_df):,} rows → {out_path}")
    return out_path


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
    parser = argparse.ArgumentParser(description="Download CropOS training data")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--start", default=None, help="Override training_start in config")
    parser.add_argument("--end", default=None, help="Override training_end in config")
    parser.add_argument(
        "--skip-push", action="store_true", help="Skip HuggingFace upload (local run)"
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.skip_push:
        raise ValueError(
            "Set HF_TOKEN environment variable (or pass --skip-push for a local-only run)"
        )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    start = args.start or cfg["training_start"]
    end = args.end or cfg["training_end"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-detect HF username from token and create repo if needed
    if not args.skip_push:
        api = HfApi()
        username = api.whoami(token=token)["name"]
        repo_id = f"{username}/{HF_DATASET_REPO_NAME}"
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            token=token,
            exist_ok=True,
        )
        logger.info(f"HF dataset repo ready: hf://datasets/{repo_id}")
    else:
        repo_id = None

    # Phase 1: ERA5 — runs alone (archive-api.open-meteo.com is rate-limited).
    # Checkpoint is pushed to HF after each batch so the next run can resume.
    results: dict[str, Path] = {}
    logger.info("Phase 1: ERA5 (archive-api.open-meteo.com — sequential, HF checkpoint)")
    results["era5"] = download_era5(cfg, start, end, outdir, repo_id=repo_id, hf_token=token)
    logger.info("✓ era5 phase complete")

    # Open-Meteo has a global per-IP rate limit across all their endpoints.
    # Let the rate limit window clear before starting NWP (also Open-Meteo).
    RATE_LIMIT_COOLDOWN_S = 65
    logger.info(
        f"Cooling down {RATE_LIMIT_COOLDOWN_S}s for Open-Meteo rate limit to reset "
        f"before Phase 2..."
    )
    time.sleep(RATE_LIMIT_COOLDOWN_S)

    # Phase 2: METAR (Iowa State) + NWP (Open-Meteo forecast API) in parallel.
    # Different servers so they can run concurrently.
    logger.info("Phase 2: METAR + NWP in parallel")
    tasks = {
        "metar": lambda: download_metar(cfg, start, end, outdir),
        "nwp": lambda: download_nwp(cfg, start, end, outdir),
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                logger.info(f"✓ {name} complete")
            except Exception as exc:
                logger.error(f"✗ {name} failed: {exc}")
                raise

    # Push all parquet files to HuggingFace Datasets
    if not args.skip_push:
        logger.info("Pushing to HuggingFace Datasets...")
        for path in results.values():
            push_to_hf(path, repo_id, token)
        logger.info(f"Done — all data at hf://datasets/{repo_id}")
    else:
        logger.info(f"Done — data saved locally to {outdir}/")


if __name__ == "__main__":
    main()
