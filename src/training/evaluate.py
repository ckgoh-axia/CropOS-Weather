"""Forecast skill metrics — Brier, CSI, FNR, Brier Skill Score."""
from __future__ import annotations

from typing import Dict

import numpy as np


def brier_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean((predicted - observed) ** 2))


def critical_success_index(
    predicted: np.ndarray, observed: np.ndarray, threshold: float = 0.5
) -> float:
    pred_bin = (predicted >= threshold).astype(int)
    tp = np.sum(pred_bin * observed)
    fp = np.sum(pred_bin * (1 - observed))
    fn = np.sum((1 - pred_bin) * observed)
    return float(tp / (tp + fp + fn + 1e-8))


def false_negative_rate(
    predicted: np.ndarray, observed: np.ndarray, threshold: float = 0.5
) -> float:
    """Fraction of actual rain events the model missed — the costly error."""
    pred_bin = (predicted >= threshold).astype(int)
    fn = np.sum((1 - pred_bin) * observed)
    total_rain = np.sum(observed)
    return float(fn / (total_rain + 1e-8))


def compute_skill_report(
    model_probs: np.ndarray,
    nwp_probs: np.ndarray,
    observed: np.ndarray,
) -> Dict[str, float]:
    """Full skill comparison: GNN vs. NWP baseline."""
    gnn_bs = brier_score(model_probs, observed)
    nwp_bs = brier_score(nwp_probs, observed)
    return {
        "gnn_brier": gnn_bs,
        "nwp_brier": nwp_bs,
        "brier_skill_score": 1.0 - (gnn_bs / (nwp_bs + 1e-8)),
        "gnn_csi": critical_success_index(model_probs, observed),
        "nwp_csi": critical_success_index(nwp_probs, observed),
        "gnn_fnr": false_negative_rate(model_probs, observed),
        "nwp_fnr": false_negative_rate(nwp_probs, observed),
    }
