# tests/unit/test_phase0_split.py
"""Regression guard for scripts/phase0_benchmark.py's _temporal_split.

Two fix rounds were spent establishing that the fit/score split for Phase 0
calibration must be a temporal (by-date) split with zero day overlap, never
a random row split — a random split leaks same-day, cross-station
correlation across the fit/score boundary and lets a near-zero-information
predictor look skilful out-of-sample (Task 4 fix-round-1 Critical 2). This
test exists only to catch a silent regression of that property; it is not a
general test suite for the script.
"""
import pandas as pd

from scripts.phase0_benchmark import _temporal_split


def _matched_df(n_days: int, per_day: int = 20) -> pd.DataFrame:
    """A frame shaped like the `sub` frames _score_cell/_score_pair pass to
    _temporal_split: one 'issuance' column plus arbitrary payload."""
    dates = pd.date_range("2024-05-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for d in dates:
        for i in range(per_day):
            rows.append({"issuance": d, "precip_mm": float(i), "era5_rain": float(i % 2)})
    return pd.DataFrame(rows)


def test_split_has_zero_day_overlap():
    sub = _matched_df(n_days=20)
    fit_df, score_df = _temporal_split(sub, fit_frac=0.7)

    fit_days = set(fit_df["issuance"].unique())
    score_days = set(score_df["issuance"].unique())

    assert fit_days, "fit half must be non-empty"
    assert score_days, "score half must be non-empty"
    assert fit_days.isdisjoint(score_days), (
        "fit and score halves share at least one issuance date — this is "
        "exactly the leakage fix-round-1 Critical 2 closed"
    )


def test_fit_max_precedes_score_min():
    sub = _matched_df(n_days=20)
    fit_df, score_df = _temporal_split(sub, fit_frac=0.7)

    fit_max = fit_df["issuance"].max()
    score_min = score_df["issuance"].min()
    assert fit_max < score_min, (
        f"fit_max ({fit_max}) must strictly precede score_min ({score_min}) "
        "— a temporal split with any inversion is not a valid guard against "
        "cross-boundary leakage"
    )


def test_split_covers_every_row_exactly_once():
    sub = _matched_df(n_days=15)
    fit_df, score_df = _temporal_split(sub, fit_frac=0.7)

    assert len(fit_df) + len(score_df) == len(sub)
    combined_idx = sorted(list(fit_df.index) + list(score_df.index))
    assert combined_idx == sorted(sub.index)


def test_too_few_dates_returns_empty_halves():
    # _temporal_split's documented degenerate case: fewer than 2 unique
    # issuance dates can't be split into a fit half and a score half at all.
    sub = _matched_df(n_days=1)
    fit_df, score_df = _temporal_split(sub)
    assert fit_df.empty
    assert score_df.empty
