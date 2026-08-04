#!/usr/bin/env python3
"""Verify the HF dataset is complete and correct — no large file downloads.

Checks:
  1. How many era5_batches/batch_NNNN.parquet files exist (expect 198)
  2. Whether era5_thailand.parquet is present and its reported size
  3. Whether era5_north.parquet exists (top-up patch file)
  4. Whether nwp_features.parquet exists and its size
  5. Whether metar_thai.parquet exists and its size

Usage:
    HF_TOKEN=<token> python scripts/verify_hf_data.py
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi

HF_DATASET_REPO_NAME = "cropos-data"
EXPECTED_BATCHES = 198
EXPECTED_GRID_PTS = 1980


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    username = api.whoami(token=token)["name"]
    repo_id = f"{username}/{HF_DATASET_REPO_NAME}"
    print(f"Checking hf://datasets/{repo_id}\n")

    repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))

    # ── ERA5 batch checkpoint files ──────────────────────────────────────────
    batch_files = sorted(
        f for f in repo_files
        if f.startswith("era5_batches/batch_") and f.endswith(".parquet")
    )
    batch_indices = set()
    for f in batch_files:
        try:
            batch_indices.add(int(f.split("batch_")[1].split(".")[0]))
        except (IndexError, ValueError):
            pass

    missing_batches = [i for i in range(EXPECTED_BATCHES) if i not in batch_indices]
    status = "✓" if len(batch_files) == EXPECTED_BATCHES else "✗"
    print(f"{status} ERA5 batch files: {len(batch_files)}/{EXPECTED_BATCHES}")
    if missing_batches:
        print(f"   MISSING batch indices: {missing_batches}")

    # ── Main assembled parquet ───────────────────────────────────────────────
    ERA5_FILE = "era5_thailand.parquet"
    if ERA5_FILE in repo_files:
        info = api.get_paths_info(repo_id, [ERA5_FILE], repo_type="dataset", token=token)
        size_mb = info[0].size / 1024 / 1024 if info else 0
        print(f"✓ {ERA5_FILE}: {size_mb:,.0f} MB")
    else:
        print(f"✗ {ERA5_FILE}: NOT FOUND")

    # ── Top-up patch file (only present if northern points were missing) ─────
    NORTH_FILE = "era5_north.parquet"
    if NORTH_FILE in repo_files:
        info = api.get_paths_info(repo_id, [NORTH_FILE], repo_type="dataset", token=token)
        size_mb = info[0].size / 1024 / 1024 if info else 0
        print(f"ℹ {NORTH_FILE}: {size_mb:,.0f} MB (top-up patch present)")
    else:
        print(f"ℹ {NORTH_FILE}: not present (expected if all 198 batches were complete)")

    # ── NWP features ────────────────────────────────────────────────────────
    NWP_FILE = "nwp_features.parquet"
    if NWP_FILE in repo_files:
        info = api.get_paths_info(repo_id, [NWP_FILE], repo_type="dataset", token=token)
        size_mb = info[0].size / 1024 / 1024 if info else 0
        print(f"✓ {NWP_FILE}: {size_mb:,.0f} MB")
    else:
        print(f"✗ {NWP_FILE}: NOT FOUND")

    # ── NWP forward-filled ───────────────────────────────────────────────────
    NWP_FFILL_FILE = "nwp_features_ffill.parquet"
    if NWP_FFILL_FILE in repo_files:
        info = api.get_paths_info(repo_id, [NWP_FFILL_FILE], repo_type="dataset", token=token)
        size_mb = info[0].size / 1024 / 1024 if info else 0
        print(f"✓ {NWP_FFILL_FILE}: {size_mb:,.0f} MB")
    else:
        print(f"✗ {NWP_FFILL_FILE}: NOT FOUND (NWP forward-fill not yet run)")

    # ── METAR ────────────────────────────────────────────────────────────────
    METAR_FILE = "metar_thai.parquet"
    if METAR_FILE in repo_files:
        info = api.get_paths_info(repo_id, [METAR_FILE], repo_type="dataset", token=token)
        size_mb = info[0].size / 1024 / 1024 if info else 0
        print(f"✓ {METAR_FILE}: {size_mb:,.0f} MB")
    else:
        print(f"✗ {METAR_FILE}: NOT FOUND")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    all_ok = (len(batch_files) == EXPECTED_BATCHES and ERA5_FILE in repo_files)
    if all_ok:
        print("✓ ERA5 data looks complete.")
        if NWP_FFILL_FILE not in repo_files:
            print("  Next: trigger NWP Forward-fill workflow.")
    else:
        print("✗ Some data is missing — review the items above.")


if __name__ == "__main__":
    main()
