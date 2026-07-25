import numpy as np
from src.training.evaluate import (
    brier_score, critical_success_index, false_negative_rate, compute_skill_report,
)


def test_brier_perfect_forecast_is_zero():
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(obs, obs) == 0.0


def test_brier_worst_forecast_is_one():
    obs = np.array([1.0, 0.0])
    pred = np.array([0.0, 1.0])
    assert brier_score(pred, obs) == 1.0


def test_csi_perfect_is_one():
    obs = np.array([1, 0, 1, 1])
    pred = np.array([0.9, 0.1, 0.9, 0.9])
    assert np.isclose(critical_success_index(pred, obs), 1.0)


def test_false_negative_rate_all_missed_is_one():
    obs = np.array([1, 1, 1])
    pred = np.array([0.1, 0.1, 0.1])
    assert np.isclose(false_negative_rate(pred, obs), 1.0)


def test_brier_skill_score_positive_when_model_beats_baseline():
    obs = np.array([1, 0, 1, 0, 1])
    good = np.array([0.9, 0.1, 0.9, 0.1, 0.9])  # model
    bad  = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # NWP baseline
    report = compute_skill_report(model_probs=good, nwp_probs=bad, observed=obs)
    assert report["brier_skill_score"] > 0.0, "Model must beat NWP baseline"
    assert report["gnn_csi"] > report["nwp_csi"]
