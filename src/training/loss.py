"""BrierCSILoss — combined Brier Score and Critical Success Index loss."""
import torch
import torch.nn as nn


class BrierCSILoss(nn.Module):
    """
    Brier Score: calibrates probability estimates. Range [0, 1], lower is better.
    CSI (Threat Score): TP/(TP+FP+FN) — rewards catching rain events without
        over-predicting. Soft version uses raw probabilities for differentiability.

    Combined: brier_weight × Brier + csi_weight × (1 - CSI)
    """

    def __init__(self, brier_weight: float = 0.7, csi_weight: float = 0.3):
        super().__init__()
        self.brier_weight = brier_weight
        self.csi_weight = csi_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (batch, n_horizons) predicted rain probabilities in [0, 1]
        target: (batch, n_horizons) binary rain labels {0, 1}
        """
        brier = torch.mean((pred - target) ** 2)
        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1 - target))
        fn = torch.sum((1 - pred) * target)
        csi = tp / (tp + fp + fn + 1e-8)
        return self.brier_weight * brier + self.csi_weight * (1.0 - csi)
