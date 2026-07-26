# tests/unit/test_evaluate.py
"""Tests for src/training/evaluate.py — Brier, CSI, FNR, skill score."""
import numpy as np
import pytest

from src.training.evaluate import (
    brier_score,
    compute_skill_report,
    critical_success_index,
    false_negative_rate,
)


# ── brier_score ───────────────────────────────────────────────────────────────

def test_brier_perfect_forecast_is_zero():
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    pred = obs.copy()
    assert brier_score(pred, obs) == pytest.approx(0.0)


def test_brier_worst_forecast_is_one():
    obs = np.array([1.0, 1.0])
    pred = np.array([0.0, 0.0])
    assert brier_score(pred, obs) == pytest.approx(1.0)


def test_brier_climatology_is_0_25():
    """Predicting 0.5 everywhere on balanced binary labels → Brier = 0.25."""
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    pred = np.full(4, 0.5)
    assert brier_score(pred, obs) == pytest.approx(0.25)


# ── critical_success_index ────────────────────────────────────────────────────

def test_csi_perfect():
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    pred = np.array([1.0, 0.0, 1.0, 0.0])
    assert critical_success_index(pred, obs) == pytest.approx(1.0, abs=1e-4)


def test_csi_zero_when_all_miss():
    obs = np.array([1.0, 1.0])
    pred = np.array([0.0, 0.0])
    assert critical_success_index(pred, obs) == pytest.approx(0.0, abs=1e-4)


def test_csi_threshold_matters():
    obs = np.array([1.0, 0.0])
    pred = np.array([0.4, 0.4])
    csi_low = critical_success_index(pred, obs, threshold=0.3)
    csi_high = critical_success_index(pred, obs, threshold=0.5)
    # At threshold 0.3 both are predicted positive → FP inflates; at 0.5 both negative
    assert csi_low != csi_high


# ── false_negative_rate ───────────────────────────────────────────────────────

def test_fnr_zero_when_all_rain_caught():
    obs = np.array([1.0, 0.0, 1.0])
    pred = np.array([1.0, 0.0, 1.0])
    assert false_negative_rate(pred, obs) == pytest.approx(0.0, abs=1e-4)


def test_fnr_one_when_all_rain_missed():
    obs = np.array([1.0, 1.0])
    pred = np.array([0.0, 0.0])
    assert false_negative_rate(pred, obs) == pytest.approx(1.0, abs=1e-4)


def test_fnr_half_when_half_missed():
    obs = np.array([1.0, 1.0, 0.0, 0.0])
    pred = np.array([1.0, 0.0, 0.0, 1.0])
    assert false_negative_rate(pred, obs) == pytest.approx(0.5, abs=1e-4)


# ── compute_skill_report ──────────────────────────────────────────────────────

def test_skill_report_has_all_keys():
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    pred = np.array([0.8, 0.2, 0.7, 0.1])
    nwp = np.array([0.6, 0.4, 0.5, 0.5])
    report = compute_skill_report(pred, nwp, obs)
    for key in ["gnn_brier", "nwp_brier", "brier_skill_score", "gnn_csi", "nwp_csi",
                "gnn_fnr", "nwp_fnr"]:
        assert key in report


def test_skill_score_positive_when_gnn_beats_nwp():
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    perfect = obs.copy()
    climatology = np.full(4, 0.5)
    report = compute_skill_report(perfect, climatology, obs)
    assert report["brier_skill_score"] > 0


def test_skill_score_negative_when_gnn_worse_than_nwp():
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    terrible = 1.0 - obs  # worst possible
    climatology = np.full(4, 0.5)
    report = compute_skill_report(terrible, climatology, obs)
    assert report["brier_skill_score"] < 0
