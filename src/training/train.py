"""Training loop for CropOSGNN with MLflow tracking.

Run
---
    python -m src.training.train                     # uses configs/ and HF data
    python -m src.training.train --local data/raw    # reads parquets from disk
    python -m src.training.train --config-dir configs --local data/raw

Environment variables
---------------------
    HF_TOKEN            — HuggingFace read token (required unless --local)
    HF_DATASET_REPO     — Override auto-detected repo id
    MLFLOW_TRACKING_URI — MLflow backend (default: local mlruns/)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.features.dataset import load_dataset_from_hf, load_dataset_from_parquets
from src.features.engineer import ERA5_FEATURE_NAMES, FeatureScaler
from src.ingestion.metar import STATION_COORDS, THAI_METAR_STATIONS
from src.models.gnn import CropOSGNN
from src.training.loss import BrierCSILoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(config_dir: str = "configs", local_data_dir: str | None = None) -> None:
    # ── load configs ──────────────────────────────────────────────────────────
    with open(f"{config_dir}/model.yaml") as f:
        mcfg = yaml.safe_load(f)
    with open(f"{config_dir}/training.yaml") as f:
        tcfg = yaml.safe_load(f)
    with open(f"{config_dir}/data.yaml") as f:
        dcfg = yaml.safe_load(f)

    gnn_cfg = mcfg["gnn"]
    horizons_h: list[int] = dcfg["forecast_horizons"]
    threshold_mm: float = dcfg["precipitation_label_threshold_mm"]
    era5_node_radius_km: float = gnn_cfg.get("edge_radius_km", 100.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── build datasets ────────────────────────────────────────────────────────
    logger.info("Building training dataset...")
    if local_data_dir:
        data_dir = Path(local_data_dir)

        # Try the 22-var NWP features file first; fall back to legacy baseline.
        nwp_candidates = ["nwp_features.parquet", "nwp_baseline.parquet"]
        nwp_path = None
        for name in nwp_candidates:
            candidate = data_dir / name
            if candidate.exists():
                nwp_path = candidate
                logger.info(f"Using NWP file: {name}")
                break
        if nwp_path is None:
            raise FileNotFoundError(
                f"No NWP parquet found in {data_dir}. "
                f"Expected one of: {nwp_candidates}"
            )

        # Include northern ERA5 top-up if present (adds grid points for Bangkok, north Thailand)
        era5_north = data_dir / "era5_north.parquet"
        era5_north_path = era5_north if era5_north.exists() else None
        if era5_north_path:
            logger.info("Found era5_north.parquet — northern grid will be merged")

        train_ds = load_dataset_from_parquets(
            era5_path=data_dir / "era5_thailand.parquet",
            nwp_path=nwp_path,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=dcfg["training_start"],
            end_date=dcfg["training_end"],
            era5_node_radius_km=era5_node_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
            era5_north_path=era5_north_path,
        )
        val_ds = load_dataset_from_parquets(
            era5_path=data_dir / "era5_thailand.parquet",
            nwp_path=nwp_path,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=dcfg["validation_start"],
            end_date=dcfg["validation_end"],
            era5_node_radius_km=era5_node_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
            era5_north_path=era5_north_path,
        )
    else:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Set HF_TOKEN or pass --local <dir>")

        from huggingface_hub import HfApi
        username = os.environ.get("HF_USER") or HfApi().whoami(token=hf_token)["name"]
        repo_id = os.environ.get("HF_DATASET_REPO") or f"{username}/cropos-data"
        logger.info(f"Loading data from hf://datasets/{repo_id}")

        train_ds = load_dataset_from_hf(
            repo_id=repo_id,
            hf_token=hf_token,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=dcfg["training_start"],
            end_date=dcfg["training_end"],
            era5_node_radius_km=era5_node_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
        )
        val_ds = load_dataset_from_hf(
            repo_id=repo_id,
            hf_token=hf_token,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=dcfg["validation_start"],
            end_date=dcfg["validation_end"],
            era5_node_radius_km=era5_node_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
        )

    logger.info(
        f"Train: {len(train_ds):,} samples  |  Val: {len(val_ds):,} samples"
    )

    # ── fit feature scalers on training data ──────────────────────────────────
    # Scalers are fit on training set ONLY; saved so inference uses identical scaling.
    # Applied in-place to both train and val lookup dicts so __getitem__ returns
    # already-scaled tensors with no overhead per batch.
    logger.info("Fitting feature scalers on training data...")
    import pandas as pd
    Path("checkpoints").mkdir(exist_ok=True)

    # ERA5 scaler — build sample from first 500 training timestamps
    era5_scaler = FeatureScaler()
    sample_ts = list(train_ds._era5_by_ts)[:500]
    era5_sample = pd.DataFrame(
        np.concatenate([train_ds._era5_by_ts[ts] for ts in sample_ts], axis=0),
        columns=train_ds.era5_feature_cols,
    )
    era5_scaler.fit(era5_sample, train_ds.era5_feature_cols)
    era5_scaler.save("checkpoints/era5_scaler.npz")

    # NWP scaler — build sample from first 500 training timestamps
    nwp_scaler = FeatureScaler()
    nwp_sample = pd.DataFrame(
        np.concatenate(
            [train_ds._nwp_by_ts[ts] for ts in sample_ts if ts in train_ds._nwp_by_ts],
            axis=0,
        ),
        columns=train_ds.nwp_feature_cols,
    )
    nwp_scaler.fit(nwp_sample, train_ds.nwp_feature_cols)
    nwp_scaler.save("checkpoints/nwp_scaler.npz")

    # Apply ERA5 scaler to BOTH train and val lookup dicts
    era5_mean = np.array(
        [era5_scaler._means[c] for c in train_ds.era5_feature_cols], dtype=np.float32
    )
    era5_std = np.array(
        [era5_scaler._stds[c] for c in train_ds.era5_feature_cols], dtype=np.float32
    )
    for ds in (train_ds, val_ds):
        for ts in ds._era5_by_ts:
            ds._era5_by_ts[ts] = (ds._era5_by_ts[ts] - era5_mean) / era5_std

    # Apply NWP scaler to BOTH train and val lookup dicts
    nwp_mean = np.array(
        [nwp_scaler._means[c] for c in train_ds.nwp_feature_cols], dtype=np.float32
    )
    nwp_std = np.array(
        [nwp_scaler._stds[c] for c in train_ds.nwp_feature_cols], dtype=np.float32
    )
    for ds in (train_ds, val_ds):
        for ts in ds._nwp_by_ts:
            ds._nwp_by_ts[ts] = (ds._nwp_by_ts[ts] - nwp_mean) / nwp_std

    logger.info("Feature scalers fitted and applied to train + val")

    # ── model, optimizer, loss ────────────────────────────────────────────────
    era5_in      = gnn_cfg.get("era5_in", len(ERA5_FEATURE_NAMES))
    station_in   = gnn_cfg.get("local_station_in", 22)

    model = CropOSGNN(
        era5_in=era5_in,
        hidden=gnn_cfg["hidden_channels"],
        n_horizons=len(horizons_h),
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        local_station_dropout=gnn_cfg.get("local_station_dropout", 0.4),
        local_station_in=station_in,
    ).to(device)
    logger.info(
        f"Model: era5_in={era5_in}, local_station_in={station_in}, "
        f"hidden={gnn_cfg['hidden_channels']}, layers={gnn_cfg['num_layers']}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = BrierCSILoss(
        brier_weight=tcfg["loss"]["brier_weight"],
        csi_weight=tcfg["loss"]["csi_weight"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0
    )
    gradient_clip = tcfg.get("gradient_clip", 1.0)

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("cropos-gnn-thai")

    with mlflow.start_run():
        mlflow.log_params({
            "era5_in":           era5_in,
            "local_station_in":  station_in,
            "hidden":            gnn_cfg["hidden_channels"],
            "num_layers":        gnn_cfg["num_layers"],
            "lr":                tcfg["learning_rate"],
            "batch_size":        tcfg["batch_size"],
            "era5_radius_km":    era5_node_radius_km,
            "loss":              "brier_csi",
            "device":            str(device),
        })

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(tcfg["epochs"]):
            # ── train ─────────────────────────────────────────────────────
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds = model(batch)          # (n_farms_in_batch, n_horizons)
                labels = batch["farm"].y      # same shape
                loss = criterion(preds, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                train_loss_sum += float(loss)
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)

            # ── validate ──────────────────────────────────────────────────
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds = model(batch)
                    labels = batch["farm"].y
                    loss = criterion(preds, labels)
                    val_loss_sum += float(loss)
                    val_batches += 1

            val_loss = val_loss_sum / max(val_batches, 1)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )
            logger.info(
                f"Epoch {epoch:03d}: train={train_loss:.4f}  val={val_loss:.4f}"
            )

            # ── early stopping + checkpoint ───────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "checkpoints/best_model.pt")
                mlflow.pytorch.log_model(model, "model")
                logger.info(f"  ✓ new best val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= tcfg["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch} (patience exhausted)")
                    break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info("Model saved to checkpoints/best_model.pt")
    logger.info("Scalers saved to checkpoints/era5_scaler.npz")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train CropOSGNN")
    parser.add_argument("--config-dir", default="configs", help="Path to configs/ directory")
    parser.add_argument(
        "--local", metavar="DIR", default=None,
        help="Read parquets from this local directory instead of HuggingFace"
    )
    args = parser.parse_args()
    train(config_dir=args.config_dir, local_data_dir=args.local)


if __name__ == "__main__":
    _cli()
