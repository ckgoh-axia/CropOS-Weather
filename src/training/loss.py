"""BrierCSILoss — combined Brier Score and Critical Success Index loss."""
import torch
import torch.nn as nn


class BrierCSILoss(nn.Module):
    """
    Brier Score: calibrates probability estimates. Range [0, 1], lower is better.
    CSI (Threat Score): TP/(TP+FP+FN) — rewards catching rain events without
        over-predicting. Soft version uses raw probabilities for differentiability.

    pos_weight: up-weight positive (rain) samples to handle class imbalance.
        Set to n_dry/n_rain (≈16.5 at 5.7% rain rate) so rain events count equally
        to dry events in the gradient signal. Default 1.0 = no weighting.

    Combined: brier_weight × Brier + csi_weight × (1 - CSI)
    """

    def __init__(
        self,
        brier_weight: float = 0.7,
        csi_weight: float = 0.3,
        pos_weight: float = 1.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.brier_weight = brier_weight
        self.csi_weight = csi_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (batch, n_horizons) predicted rain probabilities in [0, 1]
        target: (batch, n_horizons) binary rain labels {0, 1}
        """
        # Weighted Brier: rain samples weighted pos_weight× higher than dry
        weights = torch.where(
            target > 0.5,
            torch.full_like(target, self.pos_weight),
            torch.ones_like(target),
        )
        brier = (weights * (pred - target) ** 2).sum() / weights.sum()

        # Soft CSI with pos_weight applied to rain terms (TP, FN)
        tp = (pred * target * self.pos_weight).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target * self.pos_weight).sum()
        csi = tp / (tp + fp + fn + self.smooth)

        return self.brier_weight * brier + self.csi_weight * (1.0 - csi)


class DualHeadLoss(nn.Module):
    """Combined loss for dual-head model (probability + mm regression).

    Probability head: BrierCSI loss.
    Regression head: Huber loss on log(mm + 1) to handle skewed precipitation.

    Args:
        brier_weight: Weight for Brier term.
        csi_weight:   Weight for CSI term.
        pos_weight:   Up-weight for rain samples (n_dry/n_rain). Default 1.0.
        reg_weight:   Weight for regression (mm) Huber loss.
        huber_delta:  Delta for Huber loss (in log space).
    """

    def __init__(
        self,
        brier_weight: float = 0.5,
        csi_weight: float = 0.3,
        pos_weight: float = 1.0,
        reg_weight: float = 0.2,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self._brier_csi = BrierCSILoss(
            brier_weight=brier_weight,
            csi_weight=csi_weight,
            pos_weight=pos_weight,
        )
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
