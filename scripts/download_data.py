#!/usr/bin/env python3
"""CLI: download all data sources for the configured date range."""
import argparse
import yaml
import logging
from pathlib import Path
from src.ingestion.metar import fetch_all_thai_stations
from src.ingestion.era5 import fetch_era5_grid, build_thailand_grid
from src.ingestion.imerg import download_imerg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CropOS training data")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    start = args.start or cfg["training_start"]
    end = args.end or cfg["training_end"]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading METAR: {start} → {end}")
    metar_df = fetch_all_thai_stations(start, end)
    metar_df.to_parquet(outdir / "metar_thai.parquet", index=False)
    logger.info(f"METAR: {len(metar_df):,} rows saved")

    logger.info("Downloading ERA5 (0.25° grid over Thailand)...")
    lat_pts, lon_pts = build_thailand_grid(spacing_deg=0.25)
    era5_df = fetch_era5_grid(lat_pts, lon_pts, start, end)
    era5_df.to_parquet(outdir / "era5_thailand.parquet", index=False)
    logger.info(f"ERA5: {len(era5_df):,} rows saved")

    logger.info("Downloading GPM IMERG...")
    download_imerg(start, end, data_dir=str(outdir / "imerg"))
    logger.info("IMERG: download complete")


if __name__ == "__main__":
    main()
