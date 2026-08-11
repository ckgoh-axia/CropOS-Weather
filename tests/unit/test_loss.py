# tests/unit/test_loss.py
"""Tests for BrierCSILoss — combined Brier + soft-CSI training loss."""
import pytest
import torch

from src.training.loss import BrierCSILoss, DualHeadLoss


def test_loss_perfect_prediction_is_near_zero():
    loss_fn = BrierCSILoss()
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = loss_fn(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_loss_worst_prediction_is_high():
    loss_fn = BrierCSILoss()
    pred = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0]])
    loss = loss_fn(pred, target)
    assert loss.item() > 0.5


def test_loss_is_scalar():
    loss_fn = BrierCSILoss()
    pred = torch.rand(4, 4)
    target = (torch.rand(4, 4) > 0.5).float()
    loss = loss_fn(pred, target)
    assert loss.shape == torch.Size([])


def test_loss_default_weights_sum_to_one():
    loss_fn = BrierCSILoss()
    assert abs(loss_fn.brier_weight + loss_fn.csi_weight - 1.0) < 1e-6


def test_loss_custom_weights():
    loss_fn = BrierCSILoss(brier_weight=0.5, csi_weight=0.5)
    assert loss_fn.brier_weight == 0.5
    assert loss_fn.csi_weight == 0.5


def test_loss_backward_works():
    """Gradients must flow through the combined loss."""
    loss_fn = BrierCSILoss()
    pred = torch.rand(3, 4, requires_grad=True)
    target = (torch.rand(3, 4) > 0.5).float()
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_loss_all_dry_still_finite():
    """All-zero target (no rain): loss should be finite and non-negative."""
    loss_fn = BrierCSILoss()
    pred = torch.rand(5, 4)
    target = torch.zeros(5, 4)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_dual_head_loss_returns_scalar():
    loss_fn = DualHeadLoss(brier_weight=0.5, csi_weight=0.3, reg_weight=0.2)
    probs = torch.rand(4, 3)
    labels = (torch.rand(4, 3) > 0.5).float()
    mm_pred = torch.rand(4, 3).abs()
    mm_true = torch.rand(4, 3).abs()
    loss = loss_fn(probs, labels, mm_pred, mm_true)
    assert loss.shape == torch.Size([])


def test_dual_head_loss_without_mm_equals_brier_csi():
    """If reg_weight=0, DualHeadLoss must match BrierCSILoss exactly."""
    brier = BrierCSILoss(brier_weight=0.7, csi_weight=0.3)
    dual = DualHeadLoss(brier_weight=0.7, csi_weight=0.3, reg_weight=0.0)
    probs = torch.rand(6, 4)
    labels = (torch.rand(6, 4) > 0.5).float()
    mm_pred = torch.zeros(6, 4)
    mm_true = torch.zeros(6, 4)
    assert abs(float(brier(probs, labels)) - float(dual(probs, labels, mm_pred, mm_true))) < 1e-5


def test_dual_head_loss_backward():
    loss_fn = DualHeadLoss(brier_weight=0.5, csi_weight=0.3, reg_weight=0.2)
    probs = torch.rand(4, 3, requires_grad=True)
    labels = (torch.rand(4, 3) > 0.5).float()
    mm_pred = torch.rand(4, 3, requires_grad=True)
    mm_true = torch.rand(4, 3).abs()
    loss = loss_fn(probs, labels, mm_pred, mm_true)
    loss.backward()
    assert probs.grad is not None
    assert mm_pred.grad is not None
