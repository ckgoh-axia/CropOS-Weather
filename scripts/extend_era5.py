#!/usr/bin/env python3
"""Extend ERA5 coverage to 2023-2026.

Uses era5_ext_batches/ checkpoint namespace on HF — does NOT conflict with the
existing era5_batches/ checkpoints from the 2015-2022 download.
Output: era5_recent.parquet on HuggingFace.

Run as a cron workflow (hourly) until all 198 batches are done — same pattern
as the original ERA5 download cron. Typically completes in ~4 hours of runs.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET_REPO_NAME = "cropos-data"
HF_BATCH_PREFIX = "era5_ext_batches"   # Separate from era5_batches/ (2015-2022)
_MAX_BATCHES_PER_RUN = 4               # Open-Meteo free tier: ~4 batches/hour


def _pull_checkpoint(checkpoint_dir: Path, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        api = HfApi()
        repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    except Exception as exc:
        logger.warning(f"Could not list HF files: {exc} — starting fresh")
        return
    batch_files = [f for f in repo_files if f.startswith(f"{HF_BATCH_PREFIX}/batch_")]
    pulled = 0
    for hf_path in batch_files:
        local = checkpoint_dir / Path(hf_path).name
        if local.exists():
            continue
        try:
            hf_hub_download(repo_id=repo_id, filename=hf_path, repo_type="dataset",
                            token=token, local_dir=str(checkpoint_dir))
            nested = checkpoint_dir / hf_path
            if nested.exists():
                nested.rename(local)
            pulled += 1
        except Exception:
            pass
    if pulled:
        logger.info(f"ERA5-EXT | restored {pulled} checkpoint files from HF")


def _push_batches(batch_files: list[Path], repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    pushed = 0
    for bf in sorted(batch_files):
        try:
            api.upload_file(
                path_or_fileobj=str(bf),
                path_in_repo=f"{HF_BATCH_PREFIX}/{bf.name}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            pushed += 1
        except Exception as exc:
            logger.warning(f"ERA5-EXT | failed to push {bf.name}: {exc}")
    if pushed:
        logger.info(f"ERA5-EXT | pushed {pushed} batch files to HF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend ERA5 to 2023-2026")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.skip_push:
        raise ValueError("Set HF_TOKEN env var (or --skip-push for local run)")

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    username = api.whoami(token=token)["name"]
    repo_id = f"{username}/{HF_DATASET_REPO_NAME}"
    if not args.skip_push:
        create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=token, exist_ok=True)

    outdir = Path("data/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / ".era5_ext_checkpoint"   # Separate local dir too

    if not args.skip_push:
        _pull_checkpoint(checkpoint_dir, repo_id, token)

    from src.ingestion.era5 import Era5PartialDownload, build_thailand_grid, fetch_era5_grid
    lat_pts, lon_pts = build_thailand_grid(spacing_deg=0.25)
    pre_existing = set(checkpoint_dir.glob("batch_*.parquet")) if checkpoint_dir.exists() else set()

    partial = False
    try:
        fetch_era5_grid(
            lat_pts, lon_pts, args.start, args.end,
            checkpoint_dir=checkpoint_dir,
            max_new_batches=_MAX_BATCHES_PER_RUN,
        )
    except Era5PartialDownload as exc:
        partial = not exc.is_complete
        logger.info(f"ERA5-EXT | {exc}")
    finally:
        if not args.skip_push:
            new_batches = sorted(set(checkpoint_dir.glob("batch_*.parquet")) - pre_existing)
            if new_batches:
                _push_batches(new_batches, repo_id, token)
            else:
                logger.info("ERA5-EXT | no new batches this run")

    if partial:
        logger.info("ERA5-EXT | partial run complete — will resume next cron tick")
        return

    # All 198 batches done — stream-assemble era5_recent.parquet (PyArrow to avoid OOM)
    import pyarrow as pa
    import pyarrow.parquet as pq

    batch_files = sorted(checkpoint_dir.glob("batch_*.parquet"))
    logger.info(f"ERA5-EXT | assembling era5_recent.parquet from {len(batch_files)} batches...")
    out_path = outdir / "era5_recent.parquet"
    writer: pq.ParquetWriter | None = None
    for bf in batch_files:
        table = pq.read_table(str(bf))
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
    if writer:
        writer.close()
    logger.info(f"ERA5-EXT | assembled → {out_path}")

    if not args.skip_push:
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="era5_recent.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        logger.info("ERA5-EXT | era5_recent.parquet pushed to HF ✓")


if __name__ == "__main__":
    main()
