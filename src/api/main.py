"""CropOS FastAPI inference service."""
from __future__ import annotations
import os
import torch
from fastapi import FastAPI
from src.api.schemas import PredictRequest, PredictResponse, HorizonForecast
from src.models.gnn import CropOSGNN
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="CropOS Precipitation Forecast API", version="0.1.0")

HORIZONS = [12, 24, 36, 48]
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
RAIN_ALERT_THRESHOLD = float(os.getenv("RAIN_ALERT_THRESHOLD", "0.5"))

_model: CropOSGNN | None = None


def _get_model() -> CropOSGNN:
    global _model
    if _model is None:
        m = CropOSGNN(era5_in=7, metar_in=5, hidden=128, n_horizons=len(HORIZONS))
        if os.path.exists(MODEL_PATH):
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"No checkpoint at {MODEL_PATH} — using random weights")
        m.eval()
        _model = m
    return _model


def run_inference(request: PredictRequest) -> torch.Tensor:
    """
    Build live graph from current ERA5 + METAR at request.timestamp,
    run GNN, return (1, n_horizons) probability tensor.

    TODO: wire up real-time ERA5 + METAR fetch here.
    For now returns placeholder output so the API contract is fully testable.
    """
    _get_model()
    return torch.rand(1, len(HORIZONS))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    probs = run_inference(request)  # (1, n_horizons)
    forecasts = [
        HorizonForecast(
            horizon_hours=h,
            rain_probability=round(float(probs[0, i]), 4),
            alert=float(probs[0, i]) >= RAIN_ALERT_THRESHOLD,
        )
        for i, h in enumerate(HORIZONS)
    ]
    return PredictResponse(
        farm_id=request.farm_id,
        forecast=forecasts,
        model_version=os.getenv("MODEL_VERSION", "dev"),
    )
