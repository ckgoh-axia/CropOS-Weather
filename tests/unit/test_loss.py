# tests/unit/test_loss.py
"""Tests for BrierCSILoss — combined Brier + soft-CSI training loss."""
import torch
import pytest

from src.training.loss import BrierCSILoss


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
