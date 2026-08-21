"""Evaluation metrics for CropOSGNN precipitation forecasting.

All functions operate on numpy arrays.  No PyTorch dependency so this module
can be used in post-hoc analysis without a GPU.

Shape conventions:
  probs  : (n_samples, n_horizons) — predicted rain probability ∈ [0, 1]
  labels : (n_samples, n_horizons) — binary rain label {0, 1}

For per-station functions:
  probs  : (n_timestamps, n_stations, n_horizons)
  labels : (n_timestamps, n_stations, n_horizons)
"""
from __future__ import annotations

import numpy as np


def brier_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean squared error between probability and binary label, per horizon.

    Args:
        probs:  (N, H) predicted probabilities.
        labels: (N, H) binary labels.
    Returns:
        (H,) Brier score per horizon (0 = perfect, 1 = worst possible).
    """
    return np.mean((probs - labels) ** 2, axis=0)


def brier_skill_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Brier Skill Score vs. climatology baseline, per horizon.

    BSS = 1 - BS_model / BS_climatology.
    BSS > 0: better than climatology.
    BSS = 0: same as always predicting the base rate.
    BSS < 0: worse than climatology (bad).

    Climatology forecast = predicting the observed rain fraction for every sample.
    """
    bs_model = brier_score(probs, labels)

    # Climatology = always predict base rate (per horizon)
    base_rate = labels.mean(axis=0, keepdims=True)  # (1, H)
    clim = np.broadcast_to(base_rate, labels.shape)
    bs_clim = brier_score(clim, labels)

    # Avoid division by zero (degenerate case: all labels are 0 or all are 1)
    bss = np.where(bs_clim > 1e-9, 1.0 - bs_model / bs_clim, 0.0)
    return bss


def brier_skill_score_vs_reference(
    probs: np.ndarray, reference: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Brier Skill Score against an arbitrary reference forecast, per horizon.

    BSS = 1 - BS_model / BS_reference.

    Unlike ``brier_skill_score``, which compares against climatology, this
    compares against a competing forecast — for CropOS, the free public GFS or
    GraphCast-GFS forecast. "Better than climatology" is a weak claim when a
    farmer can already download a real forecast for nothing; this is the
    number that supports the product claim (spec §10).

    Returns 0.0 for any horizon where the reference is already perfect, since
    no improvement is possible there.
    """
    bs_model = brier_score(probs, labels)
    bs_ref = brier_score(reference, labels)
    return np.where(bs_ref > 1e-9, 1.0 - bs_model / bs_ref, 0.0)


def binary_confusion_stats(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute TP/FP/TN/FN and derived metrics at a given probability threshold.

    Args:
        probs:     (N,) predicted probabilities (flat, single horizon).
        labels:    (N,) binary labels.
        threshold: Decision boundary (default 0.5).
    Returns:
        Dict with: tp, fp, tn, fn, precision, recall, f1, csi, far, pod, miss_rate.

    Definitions important for farming:
        pod (Probability of Detection / recall): fraction of actual rain events
            that were correctly predicted. Low pod = silent misses (farmers
            apply pesticide, then it rains and washes it off).
        far (False Alarm Ratio): fraction of rain predictions that were wrong.
            High far = unnecessary delays to field operations.
        miss_rate = 1 - pod: fraction of rain events we missed entirely.
        csi (Critical Success Index): hits / (hits + misses + false alarms).
    """
    preds = (probs >= threshold).astype(np.float32)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)   # = pod
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    csi = tp / (tp + fp + fn + 1e-9)
    far = fp / (tp + fp + 1e-9)
    miss_rate = fn / (tp + fn + 1e-9)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,       # = pod
        "f1": f1,
        "csi": csi,
        "far": far,
        "pod": recall,
        "miss_rate": miss_rate,
    }


def roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC for a single horizon. Returns 0.5 if all labels are same class."""
    from sklearn.metrics import roc_auc_score
    if labels.std() < 1e-9:
        return 0.5
    return float(roc_auc_score(labels, probs))


def per_horizon_report(
    probs: np.ndarray,
    labels: np.ndarray,
    horizons_h: list[int],
    threshold: float = 0.5,
) -> dict[int, dict[str, float]]:
    """Full metric report keyed by horizon (hours).

    Args:
        probs:      (N, H) probability predictions.
        labels:     (N, H) binary labels.
        horizons_h: List of horizon values (e.g. [12, 24, 36, 48]).
        threshold:  Decision boundary for binary classification metrics.
    Returns:
        {horizon_h: {metric_name: value}}.
    """
    bs = brier_score(probs, labels)
    bss = brier_skill_score(probs, labels)

    report: dict[int, dict[str, float]] = {}
    for h_idx, h in enumerate(horizons_h):
        p = probs[:, h_idx]
        lbl = labels[:, h_idx]
        conf = binary_confusion_stats(p, lbl, threshold=threshold)
        report[h] = {
            "brier": float(bs[h_idx]),
            "bss": float(bss[h_idx]),
            "precision": conf["precision"],
            "recall": conf["recall"],
            "pod": conf["pod"],
            "f1": conf["f1"],
            "csi": conf["csi"],
            "far": conf["far"],
            "miss_rate": conf["miss_rate"],
            "tp": conf["tp"],
            "fp": conf["fp"],
            "tn": conf["tn"],
            "fn": conf["fn"],
            "auc": roc_auc(p, lbl),
            "rain_frac": float(lbl.mean()),
            "pred_mean": float(p.mean()),
        }
    return report


def per_station_report(
    probs: np.ndarray,
    labels: np.ndarray,
    station_names: list[str],
    horizons_h: list[int],
    threshold: float = 0.5,
) -> dict[str, dict[int, dict[str, float]]]:
    """Per-station, per-horizon metrics.

    Args:
        probs:         (N_ts, N_stations, H) probability predictions.
        labels:        (N_ts, N_stations, H) binary labels.
        station_names: Ordered list of station IDs.
        horizons_h:    Horizon list.
        threshold:     Decision threshold.
    Returns:
        {station_id: {horizon_h: {metric: value}}}.
    """
    report: dict[str, dict[int, dict[str, float]]] = {}
    for s_idx, station in enumerate(station_names):
        report[station] = per_horizon_report(
            probs[:, s_idx, :],
            labels[:, s_idx, :],
            horizons_h=horizons_h,
            threshold=threshold,
        )
    return report


def mm_regression_metrics(
    mm_pred: np.ndarray,
    mm_true: np.ndarray,
    horizons_h: list[int],
) -> dict[int, dict[str, float]]:
    """Regression metrics for mm prediction, per horizon.

    Args:
        mm_pred:    (N, H) predicted mm values.
        mm_true:    (N, H) actual mm values.
        horizons_h: Horizon list.
    Returns:
        {horizon_h: {rmse, mae, bias}}.
    """
    report: dict[int, dict[str, float]] = {}
    for h_idx, h in enumerate(horizons_h):
        pred = mm_pred[:, h_idx]
        true = mm_true[:, h_idx]
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mae = float(np.mean(np.abs(pred - true)))
        bias = float(np.mean(pred - true))
        report[h] = {"rmse": rmse, "mae": mae, "bias": bias}
    return report
