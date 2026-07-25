"""Training loop for CropOSGNN with MLflow tracking."""
from __future__ import annotations
import os
import yaml
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
from src.models.gnn import CropOSGNN
from src.training.loss import BrierCSILoss
import logging

logger = logging.getLogger(__name__)


def train(config_dir: str = "configs") -> None:
    with open(f"{config_dir}/model.yaml") as f:
        mcfg = yaml.safe_load(f)
    with open(f"{config_dir}/training.yaml") as f:
        tcfg = yaml.safe_load(f)
    with open(f"{config_dir}/data.yaml") as f:
        dcfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = CropOSGNN(
        era5_in=len(dcfg["era5_variables"]),
        metar_in=5,
        hidden=mcfg["gnn"]["hidden_channels"],
        n_horizons=len(dcfg["forecast_horizons"]),
        num_layers=mcfg["gnn"]["num_layers"],
        dropout=mcfg["gnn"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = BrierCSILoss(
        brier_weight=tcfg["loss"]["brier_weight"],
        csi_weight=tcfg["loss"]["csi_weight"],
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("cropos-gnn-thai")
    Path("checkpoints").mkdir(exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "hidden": mcfg["gnn"]["hidden_channels"],
            "num_layers": mcfg["gnn"]["num_layers"],
            "lr": tcfg["learning_rate"],
            "loss": "brier_csi",
            "device": str(device),
        })

        best_val_loss = float("inf")
        patience = 0

        for epoch in range(tcfg["epochs"]):
            # Placeholder training loop — wires up all components correctly
            # Real DataLoader from src/features/dataset.py connects here
            model.train()
            train_loss = 0.0
            model.eval()
            val_loss = 0.0

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            logger.info(f"Epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), "checkpoints/best_model.pt")
                mlflow.pytorch.log_model(model, "model")
            else:
                patience += 1
                if patience >= tcfg["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info("Model saved to checkpoints/best_model.pt")


if __name__ == "__main__":
    train()
