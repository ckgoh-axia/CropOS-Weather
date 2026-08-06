#!/usr/bin/env python3
"""Extend NWP features to 2023-2026.

Uses nwp_ext_checkpoints/ checkpoint namespace on HF — does NOT conflict with the
existing nwp_checkpoints/ from the 2016-2022 download.
Output: nwp_recent.parquet on HuggingFace.

One-shot run (not cron-based): 16 stations × ~3yr × hourly ≈ ~400k rows.
Expected runtime: 30-60 minutes. Per-station checkpoints allow resume on failure.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET_REPO_NAME = "cropos-data"
HF_CHECKPOINT_PREFIX = "nwp_ext_checkpoints"   # Separate from nwp_checkpoints/ (2016-2022)


def _pull_checkpoints(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    from huggingface_hub import hf_hub_download
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
                filename=f"{HF_CHECKPOINT_PREFIX}/{station}.parquet",
                repo_type="dataset",
                token=token,
                local_dir=str(checkpoint_dir),
            )
            nested = checkpoint_dir / HF_CHECKPOINT_PREFIX / f"{station}.parquet"
            if nested.exists():
                nested.rename(local)
            pulled += 1
        except Exception:
            pass
    if pulled:
        logger.info(f"NWP-EXT | restored {pulled} station checkpoints from HF")


def _push_checkpoints(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    pushed = 0
    for ckpt in sorted(checkpoint_dir.glob("*.parquet")):
        try:
            api.upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=f"{HF_CHECKPOINT_PREFIX}/{ckpt.name}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            pushed += 1
        except Exception as exc:
            logger.warning(f"NWP-EXT | failed to push checkpoint {ckpt.name}: {exc}")
    if pushed:
        logger.info(f"NWP-EXT | pushed {pushed} station checkpoints to HF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend NWP features to 2023-2026")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.skip_push:
        raise ValueError("Set HF_TOKEN env var (or --skip-push for local run)")

    import yaml
    with open("configs/data.yaml") as f:
        cfg = yaml.safe_load(f)

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    username = api.whoami(token=token)["name"]
    repo_id = f"{username}/{HF_DATASET_REPO_NAME}"
    if not args.skip_push:
        create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=token, exist_ok=True)

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / ".nwp_ext_checkpoint"   # Separate local dir

    if not args.skip_push:
        _pull_checkpoints(checkpoint_dir, repo_id, token)

    from src.ingestion.metar import STATION_COORDS
    from src.ingestion.nwp_baseline import fetch_all_stations

    # The historical-forecast-api only covers from 2016-01-01
    nwp_min = cfg.get("nwp_min_start", "2016-01-01")
    effective_start = args.start if args.start >= nwp_min else nwp_min
    if effective_start != args.start:
        logger.info(f"NWP-EXT | clamping start {args.start} → {effective_start} (API limit)")

    variables = cfg.get("nwp_variables", None)
    logger.info(
        f"NWP-EXT | fetching {len(STATION_COORDS)} stations, "
        f"{effective_start} → {args.end}, "
        f"{len(variables) if variables else 'default'} variables"
    )

    try:
        nwp_df = fetch_all_stations(
            STATION_COORDS, effective_start, args.end,
            variables=variables,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        if not args.skip_push:
            _push_checkpoints(checkpoint_dir, repo_id, token)

    out_path = outdir / "nwp_recent.parquet"
    nwp_df.to_parquet(out_path, index=False)
    logger.info(f"NWP-EXT | {len(nwp_df):,} rows → {out_path}")

    if not args.skip_push:
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="nwp_recent.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        logger.info("NWP-EXT | nwp_recent.parquet pushed to HF ✓")


if __name__ == "__main__":
    main()
