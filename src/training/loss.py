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


class DualHeadLoss(nn.Module):
    """Combined loss for dual-head model (probability + mm regression).

    Probability head: BrierCSI loss.
    Regression head: Huber loss on log(mm + 1) to handle skewed precipitation.

    Args:
        brier_weight: Weight for Brier term.
        csi_weight:   Weight for CSI term.
        reg_weight:   Weight for regression (mm) Huber loss.
        huber_delta:  Delta for Huber loss (in log space).
    """

    def __init__(
        self,
        brier_weight: float = 0.5,
        csi_weight: float = 0.3,
        reg_weight: float = 0.2,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self._brier_csi = BrierCSILoss(brier_weight=brier_weight, csi_weight=csi_weight)
        self.reg_weight = reg_weight
        self._huber = nn.HuberLoss(delta=huber_delta, reduction="mean")

    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        mm_pred: torch.Tensor,
        mm_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            probs:   (N, H) probability predictions.
            labels:  (N, H) binary labels.
            mm_pred: (N, H) predicted precipitation in mm (non-negative).
            mm_true: (N, H) actual precipitation in mm (from ERA5/METAR).
        Returns:
            Scalar combined loss.
        """
        cls_loss = self._brier_csi(probs, labels)
        if self.reg_weight == 0.0:
            return cls_loss
        reg_loss = self._huber(
            torch.log1p(mm_pred.clamp(min=0)),
            torch.log1p(mm_true.clamp(min=0)),
        )
        return cls_loss + self.reg_weight * reg_loss
