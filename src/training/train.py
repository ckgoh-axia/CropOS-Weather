"""Training loop for CropOSGNN with W&B + MLflow tracking.

Run
---
    python -m src.training.train                     # uses configs/ and HF data
    python -m src.training.train --local data/raw    # reads parquets from disk
    python -m src.training.train --config-dir configs --local data/raw

Environment variables
---------------------
    HF_TOKEN            — HuggingFace read token (required unless --local)
    HF_DATASET_REPO     — Override auto-detected repo id
    HF_MODEL_REPO       — HuggingFace model repo to upload checkpoint to
                          (e.g. gjmck78/cropos-gnn). Created if it doesn't exist.
                          Skipped if unset or HF_TOKEN not present.
    WANDB_API_KEY       — Weights & Biases key (optional; skipped if absent)
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
from src.training.loss import BrierCSILoss, DualHeadLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _upload_checkpoint_to_hf(
    checkpoint_dir: str,
    hf_token: str | None,
    model_repo: str | None,
) -> None:
    """Upload best_model.pt and scalers to HuggingFace model repo.

    Non-fatal — logs warnings and returns on any error so training completion
    is never blocked by a failed upload.
    """
    if not hf_token:
        print("HF_TOKEN not set — skipping HuggingFace checkpoint upload", flush=True)
        return
    if not model_repo:
        print("HF_MODEL_REPO not set — skipping HuggingFace checkpoint upload", flush=True)
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"  HF repo create warning (non-fatal): {e}", flush=True)

        # Upload checkpoint files
        files_to_upload = [
            "best_model.pt",
            "era5_scaler.npz",
            "metar_scaler.npz",
        ]
        for fname in files_to_upload:
            fpath = Path(checkpoint_dir) / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=model_repo,
                    repo_type="model",
                )
                print(f"  ✓ Uploaded {fname} → hf://{model_repo}/{fname}", flush=True)
            else:
                print(f"  WARNING: {fpath} not found, skipping upload", flush=True)

        print(
            f"Checkpoint uploaded to https://huggingface.co/{model_repo}",
            flush=True,
        )
    except Exception as exc:
        print(f"WARNING: HuggingFace upload failed (non-fatal): {exc}", flush=True)


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

        # Include northern ERA5 top-up if present (adds grid points for Bangkok, north Thailand)
        era5_north = data_dir / "era5_north.parquet"
        era5_north_path = era5_north if era5_north.exists() else None
        if era5_north_path:
            logger.info("Found era5_north.parquet — northern grid will be merged")

        # Recent ERA5 covers validation period (e.g. 2023) not present in base parquets
        era5_recent = data_dir / "era5_recent.parquet"
        era5_recent_path = era5_recent if era5_recent.exists() else None
        if era5_recent_path:
            logger.info(
                "Found era5_recent.parquet — recent ERA5 timestamps will be merged into val"
            )
        else:
            logger.warning(
                "era5_recent.parquet not found in data dir — val ERA5 will likely be empty. "
                "Run the ERA5 recent download workflow to populate it."
            )

        # METAR observations — the actual local_station (metar node) features.
        # metar_thai.parquet covers the full historical range (2015–present from Iowa State ASOS).
        metar_path = data_dir / "metar_thai.parquet"
        if not metar_path.exists():
            raise FileNotFoundError(
                f"metar_thai.parquet not found in {data_dir}. "
                f"Run the METAR ingestion workflow to download it."
            )

        train_ds = load_dataset_from_parquets(
            era5_path=data_dir / "era5_thailand.parquet",
            metar_path=metar_path,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=dcfg["training_start"],
            end_date=dcfg["training_end"],
            era5_node_radius_km=era5_node_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
            era5_north_path=era5_north_path,
            era5_recent_path=era5_recent_path,  # needed for 2023-2024 training data
        )
        val_ds = load_dataset_from_parquets(
            era5_path=data_dir / "era5_thailand.parquet",
            metar_path=metar_path,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=dcfg["validation_start"],
            end_date=dcfg["validation_end"],
            era5_node_radius_km=era5_node_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
            era5_north_path=era5_north_path,
            era5_recent_path=era5_recent_path,
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

    # Prompt GC between dataset constructions so the large ERA5 DataFrame from
    # the training set is freed before the validation set's parquet is loaded.
    import gc as _gc
    _gc.collect()

    logger.info(
        f"Train: {len(train_ds):,} samples  |  Val: {len(val_ds):,} samples"
    )
    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is EMPTY. ERA5 and METAR timestamps do not overlap "
            "for the training split. Check that metar_thai.parquet covers the "
            f"training period ({dcfg['training_start']} – {dcfg['training_end']}) "
            "and that era5_thailand.parquet has matching UTC timestamps."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset is EMPTY. ERA5 and METAR timestamps do not overlap "
            "for the validation split. Check that era5_recent.parquet covers "
            f"({dcfg['validation_start']} – {dcfg['validation_end']}) and that "
            "METAR data exists for this period. era5_recent.parquet may be "
            "incomplete — re-run the ERA5 Extend workflow if needed."
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

    # METAR scaler — fit on training data, 9 fixed features
    metar_scaler = FeatureScaler()
    metar_sample = pd.DataFrame(
        np.concatenate(
            [train_ds._metar_by_ts[ts] for ts in sample_ts if ts in train_ds._metar_by_ts],
            axis=0,
        ),
        columns=train_ds.metar_feature_cols,
    )
    metar_scaler.fit(metar_sample, train_ds.metar_feature_cols)
    metar_scaler.save("checkpoints/metar_scaler.npz")

    # Apply ERA5 scaler to BOTH train and val lookup dicts.
    # ERA5 feature columns are deterministic (ERA5_SURFACE_VARS + 4 temporal)
    # so train and val always share the same ordered column list.
    era5_mean = np.array(
        [era5_scaler._means[c] for c in train_ds.era5_feature_cols], dtype=np.float32
    )
    era5_std = np.maximum(
        np.array([era5_scaler._stds[c] for c in train_ds.era5_feature_cols], dtype=np.float32),
        1e-8,
    )
    for ds in (train_ds, val_ds):
        for ts in ds._era5_by_ts:
            ds._era5_by_ts[ts] = (ds._era5_by_ts[ts] - era5_mean) / era5_std

    # Apply METAR scaler to BOTH train and val lookup dicts.
    # METAR feature columns are fixed (METAR_FEATURE_COLS) so train and val always
    # share the same column list and ordering — no column mismatch risk.
    metar_mean = np.array(
        [metar_scaler._means[c] for c in train_ds.metar_feature_cols], dtype=np.float32
    )
    metar_std = np.maximum(
        np.array([metar_scaler._stds[c] for c in train_ds.metar_feature_cols], dtype=np.float32),
        1e-8,
    )
    for ds in (train_ds, val_ds):
        for ts in ds._metar_by_ts:
            ds._metar_by_ts[ts] = (ds._metar_by_ts[ts] - metar_mean) / metar_std

    logger.info("Feature scalers fitted and applied to train + val")

    # ── model, optimizer, loss ────────────────────────────────────────────────
    era5_in   = gnn_cfg.get("era5_in", len(ERA5_FEATURE_NAMES))
    metar_in  = gnn_cfg.get("metar_in", 9)

    dual_head: bool = gnn_cfg.get("dual_head", False)
    model = CropOSGNN(
        era5_in=era5_in,
        hidden=gnn_cfg["hidden_channels"],
        n_horizons=len(horizons_h),
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        metar_dropout=gnn_cfg.get("metar_dropout", 0.4),
        metar_in=metar_in,
        dual_head=dual_head,
    ).to(device)
    logger.info(
        f"Model: era5_in={era5_in}, metar_in={metar_in}, "
        f"hidden={gnn_cfg['hidden_channels']}, layers={gnn_cfg['num_layers']}, dual_head={dual_head}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    loss_cfg = tcfg.get("loss", {})
    pos_weight: float = loss_cfg.get("pos_weight", 1.0)
    reg_weight = loss_cfg.get("reg_weight", 0.0) if dual_head else 0.0
    if dual_head and reg_weight > 0:
        criterion = DualHeadLoss(
            brier_weight=loss_cfg.get("brier_weight", 0.5),
            csi_weight=loss_cfg.get("csi_weight", 0.3),
            reg_weight=reg_weight,
            pos_weight=pos_weight,
        )
    else:
        criterion = BrierCSILoss(
            brier_weight=loss_cfg.get("brier_weight", 0.7),
            csi_weight=loss_cfg.get("csi_weight", 0.3),
            pos_weight=pos_weight,
        )
    logger.info(
        f"Loss: brier_weight={loss_cfg.get('brier_weight', 0.7)}, "
        f"csi_weight={loss_cfg.get('csi_weight', 0.3)}, pos_weight={pos_weight}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0
    )
    gradient_clip = tcfg.get("gradient_clip", 1.0)

    # ── W&B init (optional — skipped if WANDB_API_KEY not set) ──────────────
    _wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            _wandb_run = wandb.init(
                project="cropos-gnn-thai",
                config={
                    "era5_in":        era5_in,
                    "metar_in":       metar_in,
                    "hidden":         gnn_cfg["hidden_channels"],
                    "num_layers":     gnn_cfg["num_layers"],
                    "lr":             tcfg["learning_rate"],
                    "batch_size":     tcfg["batch_size"],
                    "era5_radius_km": era5_node_radius_km,
                    "loss":           "brier_csi",
                    "pos_weight":     pos_weight,
                    "device":         str(device),
                },
                # Alert on run completion so you're notified without staying connected
                settings=wandb.Settings(
                    _save_requirements=False,
                ),
            )
            print(f"W&B run: {_wandb_run.url}", flush=True)
        except Exception as _wb_exc:
            print(f"W&B init failed (non-fatal): {_wb_exc}", flush=True)
            _wandb_run = None
    else:
        print("WANDB_API_KEY not set — skipping W&B logging", flush=True)

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("cropos-gnn-thai")

    with mlflow.start_run():
        mlflow.log_params({
            "era5_in":        era5_in,
            "metar_in":       metar_in,
            "hidden":         gnn_cfg["hidden_channels"],
            "num_layers":     gnn_cfg["num_layers"],
            "lr":             tcfg["learning_rate"],
            "batch_size":     tcfg["batch_size"],
            "era5_radius_km": era5_node_radius_km,
            "loss":           "brier_csi",
            "pos_weight":     pos_weight,
            "device":         str(device),
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
                out = model(batch)
                labels = batch["farm"].y      # (n_farms_in_batch, n_horizons)
                if dual_head and isinstance(out, tuple):
                    probs, mm_pred = out
                    mm_true = (
                        batch["farm"].precip_mm
                        if hasattr(batch["farm"], "precip_mm")
                        else torch.zeros_like(probs)
                    )
                    loss = criterion(probs, labels, mm_pred, mm_true)
                else:
                    probs = out
                    loss = criterion(probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                train_loss_sum += float(loss)
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)

            # ── validate ──────────────────────────────────────────────────
            model.eval()
            val_loss_sum = 0.0
            val_mm_mae_sum = 0.0
            val_batches = 0
            _val_preds_all: list[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    labels = batch["farm"].y
                    if dual_head and isinstance(out, tuple):
                        probs, mm_pred = out
                        mm_true = (
                            batch["farm"].precip_mm
                            if hasattr(batch["farm"], "precip_mm")
                            else torch.zeros_like(probs)
                        )
                        loss = criterion(probs, labels, mm_pred, mm_true)
                        val_mm_mae_sum += float(torch.mean(torch.abs(mm_pred - mm_true)))
                    else:
                        probs = out
                        loss = criterion(probs, labels)
                    val_loss_sum += float(loss)
                    val_batches += 1
                    _val_preds_all.append(probs.cpu().float().numpy())

            val_loss = val_loss_sum / max(val_batches, 1)
            if dual_head:
                val_mm_mae = val_mm_mae_sum / max(val_batches, 1)
            else:
                val_mm_mae = 0.0

            # Log val prediction distribution every epoch to detect mode collapse
            # (std ≈ 0 means model predicts a constant regardless of input → flat val_loss)
            if _val_preds_all:
                _preds_cat = np.concatenate(_val_preds_all, axis=0)
                _pred_mean = float(_preds_cat.mean())
                _pred_std  = float(_preds_cat.std())
                print(
                    f"  val preds: mean={_pred_mean:.4f}  std={_pred_std:.4f}  "
                    f"min={float(_preds_cat.min()):.4f}  max={float(_preds_cat.max()):.4f}",
                    flush=True,
                )
                if _pred_std < 0.005:
                    print(
                        f"  WARNING val pred std={_pred_std:.5f} — model predicting "
                        f"constant value. Check METAR data quality and ERA5 coverage.",
                        flush=True,
                    )

            metrics_to_log = {"train_loss": train_loss, "val_loss": val_loss}
            if dual_head:
                metrics_to_log["val_mm_mae"] = val_mm_mae
            mlflow.log_metrics(metrics_to_log, step=epoch)
            if _wandb_run is not None:
                wandb_metrics = {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}
                if dual_head:
                    wandb_metrics["val_mm_mae"] = val_mm_mae
                _wandb_run.log(wandb_metrics, step=epoch)
            if dual_head:
                print(
                    f"Epoch {epoch:03d}: train={train_loss:.4f}  "
                    f"val={val_loss:.4f}  val_mm_mae={val_mm_mae:.3f}",
                    flush=True,
                )
            else:
                print(f"Epoch {epoch:03d}: train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

            # ── early stopping + checkpoint ───────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "checkpoints/best_model.pt")
                try:
                    mlflow.pytorch.log_model(model, "model")
                except Exception as _mlf_exc:
                    print(
                        f"  mlflow.pytorch.log_model failed (non-fatal): {_mlf_exc}",
                        flush=True,
                    )
                print(f"  ✓ new best val loss: {val_loss:.4f}", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= tcfg["early_stopping_patience"]:
                    print(
                        f"Early stopping at epoch {epoch} (patience exhausted)",
                        flush=True,
                    )
                    break

    print(f"Training complete. Best val loss: {best_val_loss:.4f}", flush=True)
    if _wandb_run is not None:
        _wandb_run.summary["best_val_loss"] = best_val_loss
        _wandb_run.finish()
    print("Model saved to checkpoints/best_model.pt", flush=True)
    print(
        "Scalers saved to checkpoints/era5_scaler.npz  checkpoints/metar_scaler.npz",
        flush=True,
    )

    # ── upload checkpoint to HuggingFace ─────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    model_repo = os.environ.get("HF_MODEL_REPO")
    if not model_repo and hf_token:
        # Auto-detect from HF_USER or whoami
        try:
            from huggingface_hub import HfApi
            username = os.environ.get("HF_USER") or HfApi().whoami(token=hf_token)["name"]
            model_repo = f"{username}/cropos-gnn"
        except Exception:
            pass
    _upload_checkpoint_to_hf("checkpoints", hf_token, model_repo)


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
