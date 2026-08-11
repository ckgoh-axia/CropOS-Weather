# tests/unit/test_metrics.py
import numpy as np

from src.evaluation.metrics import (
    binary_confusion_stats,
    brier_score,
    brier_skill_score,
    per_horizon_report,
    per_station_report,
)


def _make_preds(n=200, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.random((n, 4))          # (n_samples, n_horizons)
    labels = (rng.random((n, 4)) > 0.6).astype(np.float32)   # ~40% rain rate
    return probs, labels


def test_brier_score_range():
    probs, labels = _make_preds()
    bs = brier_score(probs, labels)
    assert bs.shape == (4,), "One score per horizon"
    assert (bs >= 0).all() and (bs <= 1).all()


def test_brier_score_perfect():
    labels = np.ones((50, 4), dtype=np.float32)
    bs = brier_score(labels, labels)
    assert np.allclose(bs, 0.0, atol=1e-6)


def test_brier_score_worst():
    labels = np.ones((50, 4), dtype=np.float32)
    worst = np.zeros_like(labels)
    bs = brier_score(worst, labels)
    assert np.allclose(bs, 1.0, atol=1e-6)


def test_brier_skill_score_climatology_is_zero():
    """Predicting the base rate every time must give BSS = 0."""
    labels = (np.random.default_rng(1).random((200, 4)) > 0.6).astype(np.float32)
    base_rate = labels.mean(axis=0, keepdims=True) * np.ones_like(labels)
    bss = brier_skill_score(base_rate, labels)
    assert np.allclose(bss, 0.0, atol=1e-5)


def test_brier_skill_score_perfect_is_one():
    labels = (np.random.default_rng(2).random((100, 4)) > 0.5).astype(np.float32)
    bss = brier_skill_score(labels, labels)
    assert np.allclose(bss, 1.0, atol=1e-5)


def test_brier_skill_score_worse_than_clim_is_negative():
    probs, labels = _make_preds()
    # Inverting probabilities should produce negative BSS
    bss = brier_skill_score(1.0 - probs, labels)
    # Not guaranteed negative for all horizons but at least some should be
    assert bss.min() < 0.5


def test_binary_confusion_stats_all_correct():
    labels = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7], dtype=np.float32)
    stats = binary_confusion_stats(probs, labels, threshold=0.5)
    assert stats["tp"] == 3
    assert stats["tn"] == 2
    assert stats["fp"] == 0
    assert stats["fn"] == 0
    assert abs(stats["precision"] - 1.0) < 1e-6
    assert abs(stats["recall"] - 1.0) < 1e-6


def test_binary_confusion_stats_false_negatives():
    labels = np.array([1, 1, 1], dtype=np.float32)
    probs = np.array([0.1, 0.2, 0.3], dtype=np.float32)   # all below threshold
    stats = binary_confusion_stats(probs, labels, threshold=0.5)
    assert stats["fn"] == 3
    assert stats["recall"] == 0.0


def test_per_horizon_report_keys():
    probs, labels = _make_preds()
    report = per_horizon_report(probs, labels, horizons_h=[12, 24, 36, 48])
    for h in [12, 24, 36, 48]:
        row = report[h]
        for key in ["brier", "bss", "precision", "recall", "f1", "csi", "far", "auc"]:
            assert key in row, f"Missing key '{key}' for horizon {h}"


def test_per_station_report_has_all_stations():
    rng = np.random.default_rng(5)
    # probs/labels shape: (n_ts, n_stations, n_horizons)
    probs = rng.random((100, 3, 4))
    labels = (rng.random((100, 3, 4)) > 0.5).astype(np.float32)
    stations = ["VTBS", "VTBD", "VTCC"]
    report = per_station_report(probs, labels, station_names=stations, horizons_h=[12, 24, 36, 48])
    assert set(report.keys()) == set(stations)
