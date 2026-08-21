#!/usr/bin/env python3
"""Phase 0 — measure the bar before building anything.

Scores the free public rain forecasts (raw GFS and operational GraphCast-GFS)
at the 16 Thai METAR stations at 24 h and 48 h lead. That score is the bar the
redesigned model must beat, and it gates the grid download.

Every forecast is fetched through select_run(), so the publication-latency
rule in spec 4.3 is enforced for the benchmark exactly as it will be at
inference. A benchmark that cheats would set the bar too high.

Forecasts are checkpointed as one parquet shard per day under
``<out>/shards/`` — written only once that day's fetches all succeed in
full, so a day with any transient failure is retried on the next run rather
than being cached as an incomplete success (see fix-round-2 NEW-2).
``<out>/forecasts.parquet`` is always rebuilt from exactly the shards that
exist for the requested ``--start``/``--end``, so it can never silently
reflect a different window than the one just requested, and coverage below
the gate's threshold is reported honestly rather than silently passing.

Usage:
    PYTHONPATH=. python scripts/phase0_benchmark.py \\
        --labels data/raw/era5_recent.parquet \\
        --metar  data/raw/metar_thai.parquet \\
        --out    data/phase0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.metrics import (
    brier_skill_score,
    brier_skill_score_vs_reference,
)
from src.ingestion.gfs_grib import bucket_start, gfs_key, graphcast_key, select_run
from src.ingestion.grib_fetch import BUCKET_GFS, BUCKET_GRAPHCAST, fetch_point_values
from src.ingestion.metar import STATION_COORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZONS_H = [24, 48]
THRESHOLD_MM = 1.0
ACCUM_WINDOW_H = 6

# Forecast row schema, shared by fetch_forecasts (writing shards) and
# _load_shards (reading/concatenating them + the empty-window fallback) so
# an all-empty day or an all-missing window never loses its column schema
# (fix-round-2 NEW-4 — a bare pd.DataFrame([]) has NO columns at all, which
# crashed downstream `fc.model` access on a fully-empty result).
FORECAST_COLS = [
    "issuance", "valid_time", "horizon_h", "model", "run", "lead_h",
    "station", "precip_mm",
]

# Fine-tune window per spec §4.7 — the boundary the calibration fit must
# never cross into validation/test data (see --allow-outside-fit-window
# below). This is a DESIGN boundary, not a data-availability one: it is NOT
# the same thing as GRAPHCAST_IDX_FROM below, and conflating the two once
# meant a real run's --start defaulted to a date GraphCast-GFS could not
# actually be fetched from.
FIT_START = pd.Timestamp("2024-02-05", tz="UTC")
FIT_END = pd.Timestamp("2025-06-30", tz="UTC")

# Earliest date GraphCast-GFS is actually byte-range fetchable. The
# graphcastgfs.YYYYMMDD/ *directories* exist from 2024-02-05 (spec §4.4),
# but the .idx sidecars fetch_point_values() requires to locate a message's
# byte range are not published until later — verified against
# forecasts_13_levels/: 20240424/00 has 0 .idx files, 20240428/00 has 40,
# 20240501/00 has 40, 20250630/18 has 65 (the GRIB data files themselves
# return HTTP 200 throughout; only the index is missing before this date).
# A real Phase 0 run should default --start to THIS, not FIT_START:
# anything in [FIT_START, GRAPHCAST_IDX_FROM) will fetch GFS fine but fail
# every GraphCast-GFS request, eroding scored coverage for no benefit.
GRAPHCAST_IDX_FROM = pd.Timestamp("2024-05-01", tz="UTC")

# Gate: the best calibrated prior must clear a small margin above sampling
# noise on OUT-OF-SAMPLE BSS (never in-sample — a 2-parameter logistic fit
# and scored on the same rows essentially always beats the in-sample base
# rate; see Task 4 fix-round-1 Critical 2) and must be fetched at >=90%
# SCORED coverage (Important 3 / fix-round-2 NEW-1 — scored, not raw
# per-model fetch coverage, or a coverage gap in one model can hide behind
# the other model's untouched 100%).
GATE_MIN_BSS_48H = 0.02
GATE_MIN_COVERAGE = 0.90

# Fraction of the fine-tune window (by issuance date, not row count) used to
# fit calibration; the remainder is the held-out out-of-sample score set.
FIT_SPLIT_FRAC = 0.7
MIN_CELL_N = 100
MIN_SPLIT_N = 50


def _apcp_pattern(lead_h: int) -> str:
    """Exact APCP descriptor for the 6-hour bucket ending at lead_h."""
    start = bucket_start(lead_h)
    if lead_h - start != ACCUM_WINDOW_H:
        raise ValueError(
            f"lead {lead_h} is not the end of a {ACCUM_WINDOW_H}h bucket "
            f"(bucket starts at {start})"
        )
    return f"APCP:surface:{start}-{lead_h} hour acc fcst"


def _write_shard(path: Path, df: pd.DataFrame) -> None:
    """Write a shard atomically (write-then-rename) so a crash mid-write
    cannot leave a half-written file that looks "done" on resume."""
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _stats_sidecar_path(shard_dir: Path, day: pd.Timestamp) -> Path:
    return shard_dir / f"{day.date()}.stats.json"


def _write_day_stats(shard_dir: Path, day: pd.Timestamp, day_stats: dict) -> None:
    """Write a day's fetch-stats sidecar atomically (write-then-rename).

    Written for every day fetch_forecasts actually processes, whether that
    day ends up complete or not, so a resumed run's tally reflects the real
    fetch history across invocations instead of resetting to 0 on every
    call (fix-round-2 NEW-3). See _aggregate_day_stats.
    """
    p = _stats_sidecar_path(shard_dir, day)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(day_stats))
    tmp.replace(p)


def fetch_forecasts(start: pd.Timestamp, end: pd.Timestamp, shard_dir: Path) -> None:
    """Fetch GFS and GraphCast-GFS accumulated precip for each day and horizon.

    Writes one parquet shard per day to ``shard_dir`` — but ONLY if that day
    is complete: every attempted (horizon, model) fetch for that day
    succeeded with a full 16-station result and zero exceptions. A day with
    any fetch exception is deliberately left without a shard, so a later run
    retries it instead of permanently caching it with fewer rows than
    expected (fix-round-2 NEW-2 — the previous version wrote the shard
    unconditionally after the day's loop, so a transient S3 failure eroded
    coverage silently and forever, since a later run would see the shard
    "exists" and never retry that day).

    A per-day stats sidecar (``<date>.stats.json``) is written alongside
    every day this function actually processes — complete or not — so the
    fetch tally in the report survives a resumed run (see
    _aggregate_day_stats, fix-round-2 NEW-3).

    A day whose shard already exists (i.e. previously completed in full) is
    skipped without any network call.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    newly_complete = 0
    newly_incomplete = 0
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        shard_path = shard_dir / f"{day.date()}.parquet"
        if shard_path.exists():
            logger.info(f"{day.date()}: shard exists (complete), skipping fetch")
            continue

        rows: list[dict] = []
        day_attempted = {"gfs": 0, "graphcast": 0}
        day_exceptions = {"gfs": 0, "graphcast": 0}
        day_skips = 0
        issuance = day  # 00Z issuance
        for h in HORIZONS_H:
            valid = issuance + pd.Timedelta(hours=h)
            try:
                run, lead = select_run(issuance, valid)
            except ValueError as exc:
                logger.warning(f"skip {issuance} h={h}: {exc}")
                day_skips += 1
                continue
            if lead - bucket_start(lead) != ACCUM_WINDOW_H:
                # select_run may return a lead that is not a 6h bucket end.
                logger.debug(f"skip {issuance} h={h}: lead {lead} not a 6h bucket end")
                day_skips += 1
                continue
            pattern = _apcp_pattern(lead)
            for model, bucket, keyfn in (
                ("gfs", BUCKET_GFS, gfs_key),
                ("graphcast", BUCKET_GRAPHCAST, graphcast_key),
            ):
                day_attempted[model] += 1
                try:
                    vals = fetch_point_values(
                        bucket, keyfn(run, lead), pattern, STATION_COORDS
                    )
                except Exception as exc:  # noqa: BLE001 — log, count, continue
                    logger.warning(f"{model} {run} f{lead:03d}: {exc}")
                    day_exceptions[model] += 1
                    continue
                for station, mm in vals.items():
                    rows.append(
                        {
                            "issuance": issuance,
                            "valid_time": valid,
                            "horizon_h": h,
                            "model": model,
                            "run": run,
                            "lead_h": lead,
                            "station": station,
                            "precip_mm": mm,
                        }
                    )

        _write_day_stats(
            shard_dir,
            day,
            {
                "attempted": day_attempted,
                "fetch_exceptions": day_exceptions,
                "run_select_skips": day_skips,
            },
        )

        expected_rows = sum(day_attempted.values()) * len(STATION_COORDS)
        total_exceptions = sum(day_exceptions.values())
        day_df = pd.DataFrame(rows, columns=FORECAST_COLS)
        if total_exceptions == 0 and len(day_df) == expected_rows:
            _write_shard(shard_path, day_df)
            newly_complete += 1
            logger.info(
                f"{day.date()}: complete, {len(day_df):,} rows written to {shard_path.name}"
            )
        else:
            newly_incomplete += 1
            logger.warning(
                f"{day.date()}: INCOMPLETE ({len(day_df)}/{expected_rows} rows, "
                f"{total_exceptions} fetch exceptions) — shard NOT written, will retry"
            )
    logger.info(
        f"fetch pass done: {newly_complete} day(s) newly completed, "
        f"{newly_incomplete} day(s) still incomplete (will retry next run)"
    )


def _aggregate_day_stats(shard_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Aggregate per-day fetch-stats sidecars for [start, end].

    Reading these from disk (rather than accumulating in memory inside a
    single fetch_forecasts call) is what makes the tally survive a resumed
    run: a day fetched in an earlier invocation still contributes its
    attempted/exception counts here instead of resetting to 0 (fix-round-2
    NEW-3). ``incomplete_days`` lists every requested day with no shard file
    — i.e. not yet fetched in full (fix-round-2 NEW-2); a missing shard is
    expected here, not a bug.
    """
    agg = {
        "run_select_skips": 0,
        "gfs": {"attempted": 0, "fetch_exceptions": 0},
        "graphcast": {"attempted": 0, "fetch_exceptions": 0},
        "incomplete_days": [],
    }
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        sidecar = _stats_sidecar_path(shard_dir, day)
        if sidecar.exists():
            day_stats = json.loads(sidecar.read_text())
            agg["run_select_skips"] += day_stats["run_select_skips"]
            for model in ("gfs", "graphcast"):
                agg[model]["attempted"] += day_stats["attempted"][model]
                agg[model]["fetch_exceptions"] += day_stats["fetch_exceptions"][model]
        shard = shard_dir / f"{day.date()}.parquet"
        if not shard.exists():
            agg["incomplete_days"].append(str(day.date()))
    return agg


def _load_shards(shard_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Concatenate the per-day shards covering [start, end] that exist.

    A day's shard exists only if fetch_forecasts completed it in full
    (fix-round-2 NEW-2): a day with any fetch exception is deliberately left
    without a shard so a later run retries it. So a missing shard here is
    expected, not an error — it means that day is not yet fully fetched
    (pending retry, or a persistent upstream failure). This function must
    not raise on it: raising would turn an ordinary transient S3 failure
    into a crashed run instead of the honest reduced-coverage FAIL the gate
    is designed to produce (see main()'s coverage/gate logic). A stale cache
    from a DIFFERENT --start/--end is still impossible here regardless of
    this relaxation: only shards for days inside the exact requested range
    are ever read, so a directory holding other date ranges' shards cannot
    leak into this result (Important 2's original concern).
    """
    frames = []
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        p = shard_dir / f"{day.date()}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=FORECAST_COLS)
    return pd.concat(frames, ignore_index=True)


def _read_era5_window(path: Path, needed_ts: set[pd.Timestamp]) -> pd.DataFrame:
    """Read only the exact ERA5 timestamps the label windows need.

    era5_recent.parquet is ~60.7M rows. A 17-month date-range filter alone
    still reads ~24M rows though only 6 of every 24 hours (the accumulation
    window for a 00Z-aligned valid time vt is [vt-5h, vt], i.e. 19:00
    through 00:00 UTC of the day before — matching GFS's own "24-30 hour
    acc fcst" style 6-hour buckets) are ever used. Filtering to the exact
    needed timestamps instead of a range narrows this to what the label
    builder actually touches (Important 4).

    NOTE: this filter is narrowing on ROWS but is not provably inert on
    label VALUES: `_build_era5_label_df` derives its nearest-grid-point-per-
    station mapping from whatever set of (lat, lon) points survives the
    filter, so a grid point that happens to be absent from the selected
    hours (but present at other hours) could in principle change which
    ERA5 cell a station is mapped to. It is not a no-op filter to reason
    about casually.
    """
    ts_list = sorted(needed_ts)
    return pq.read_table(
        path,
        columns=["timestamp", "lat", "lon", "precipitation"],
        filters=[("timestamp", "in", ts_list)],
    ).to_pandas()


def build_labels(
    era5_path: Path, metar_path: Path, valid_times: pd.Series
) -> tuple[pd.DataFrame, dict]:
    """Build 6-hour-window rain labels from ERA5 and METAR.

    Returns (labels_df, diagnostics).

    labels_df has one row per (valid_time, station):
        era5_rain   — 1.0 if the 6-hour accumulated total is >= THRESHOLD_MM,
                      0.0 if all 6 hours are present and below threshold,
                      NaN if ANY hour in the window is missing. A partial-
                      window sum would silently under-count rain (Important
                      1), so a window needs full coverage to score either
                      way.
        metar_rain  — 1.0 if ANY observed hour in the window reports
                      rain_event, 0.0 if every observed hour reports no
                      rain_event, NaN if no hour was observed at all. This
                      is occurrence-only (present-weather codes RA/TS/SH,
                      no accumulated depth) and is NOT the same event as
                      era5_rain's >=1.0mm threshold — see the report's
                      caveat before comparing the two columns.
        metar_valid — whether metar_rain is determinate (station reported
                      at least one hour in the window).

    diagnostics: {"era5_windows_total", "era5_windows_dropped_incomplete",
                  "era5_raw_null_precip_rows"}

    Note: METAR's precip_mm is deliberately not used. The Thai ASOS feed has
    no p01i column, so upstream (src/ingestion/metar.py) precip_mm is filled
    with the pd.to_numeric(..., 0) default and is identically zero across
    the entire archive — verified independently (1,034,434 rows, min/max/sum
    all 0.0). Fitting a logistic calibration against it raises immediately
    on a single-class label and previously crashed a multi-hour run with
    nothing written (Task 4 fix-round-1 Critical 1).

    CRITICAL — a null-but-present ERA5 row must never read as "0.0 mm,
    complete". `_build_era5_label_df` (src/features/dataset.py, shared with
    CropOSDataset and therefore out of scope to change here) does
    `pd.to_numeric(..., errors="coerce").fillna(0.0)` on `precip_mm`. That
    turns a genuinely-missing observation into a clean-looking dry reading
    BEFORE this function's own "any hour missing -> NaN -> drop the window"
    check ever sees it — that check only catches a grid point that is
    wholly ABSENT from the frame, not one that is present with a null
    value, so `era5_windows_dropped_incomplete` would silently read 0 even
    if every row were null. We therefore check the RAW frame here, before
    it reaches `_build_era5_label_df`, and refuse to proceed if any row is
    null (fail loudly per spec §9's "assert non-null fraction ... and fail
    loudly", rather than attempt to re-derive per-station-window dropping
    outside dataset.py's nearest-grid-point mapping, which would duplicate
    and risk diverging from that logic). See report.md / gate.json's
    `era5_raw_null_precip_rows` — it is always 0 by the time a report is
    written, because a nonzero count halts the run right here.
    """
    from src.features.dataset import _build_era5_label_df

    offsets = [pd.Timedelta(hours=k) for k in range(ACCUM_WINDOW_H)]
    unique_vt = sorted(set(valid_times))
    needed_ts = {vt - o for vt in unique_vt for o in offsets}

    era5 = _read_era5_window(era5_path, needed_ts)
    n_null = int(era5["precipitation"].isna().sum())
    if n_null > 0:
        raise SystemExit(
            f"REFUSING TO PROCEED: {n_null:,} of {len(era5):,} raw ERA5 rows "
            "in the needed window have a null 'precipitation' value. "
            "_build_era5_label_df's fillna(0.0) would silently convert "
            "these to 0.0 mm labels marked complete — exactly the failure "
            "this benchmark exists to prevent (spec §4.3 / this project's "
            "prior five-year null-to-zero incident). Investigate the ERA5 "
            "source parquet for the affected timestamps before re-running "
            "Phase 0. This is not a per-window drop — the run has stopped "
            "entirely so no gate.json/report.md is written from tainted data."
        )
    era5_lbl = _build_era5_label_df(era5, STATION_COORDS, list(STATION_COORDS))
    era5_lbl = era5_lbl.set_index(["timestamp", "station"]).sort_index()

    metar = pd.read_parquet(metar_path, columns=["timestamp", "station", "rain_event"])
    metar["timestamp"] = pd.to_datetime(metar["timestamp"], utc=True)
    # rain_event may be bool or float in the source — coerce to a numeric
    # 0/1/NaN so np.nanmax below works uniformly.
    metar["rain_event"] = pd.to_numeric(metar["rain_event"], errors="coerce")
    metar_lbl = metar.set_index(["timestamp", "station"])["rain_event"].sort_index()

    rows: list[dict] = []
    era5_windows_total = 0
    era5_windows_dropped = 0
    for vt in unique_vt:
        window = [vt - o for o in offsets]
        for station in STATION_COORDS:
            era5_windows_total += 1
            e_vals = [era5_lbl["precip_mm"].get((w, station), np.nan) for w in window]
            if any(np.isnan(v) for v in e_vals):
                era5_rain = np.nan
                era5_windows_dropped += 1
            else:
                era5_rain = float(sum(e_vals) >= THRESHOLD_MM)

            m_vals = [metar_lbl.get((w, station), np.nan) for w in window]
            m_obs = not all(np.isnan(v) for v in m_vals)
            metar_rain = float(np.nanmax(m_vals) > 0) if m_obs else np.nan

            rows.append(
                {
                    "valid_time": vt,
                    "station": station,
                    "era5_rain": era5_rain,
                    "metar_rain": metar_rain,
                    "metar_valid": bool(m_obs),
                }
            )

    diagnostics = {
        "era5_windows_total": era5_windows_total,
        "era5_windows_dropped_incomplete": era5_windows_dropped,
        # Always 0 here — see the docstring: a nonzero raw-null count halts
        # the run above, before this point is ever reached. Recorded anyway
        # so report.md/gate.json show the check ran rather than omit it.
        "era5_raw_null_precip_rows": n_null,
    }
    return pd.DataFrame(rows), diagnostics


def fit_calibration(precip_mm: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit p(rain) = sigmoid(a * log1p(mm) + b). Returns (a, b).

    This is the same functional form the residual head's prior uses (spec
    3.4), so a and b transfer directly as its initialisation.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.log1p(np.clip(precip_mm, 0, None)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=1000).fit(x, labels.astype(int))
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def apply_calibration(precip_mm: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.log1p(np.clip(precip_mm, 0, None)) + b
    return 1.0 / (1.0 + np.exp(-z))


def _safe_fit_calibration(
    precip_mm: np.ndarray, labels: np.ndarray, cell_name: str
) -> tuple[float, float] | None:
    """fit_calibration guarded against single-class inputs.

    LogisticRegression raises ValueError on a single-class y. Losing a
    multi-hour run to an unguarded fit deep in a report loop is exactly the
    failure mode this guards against (Task 4 fix-round-1 Critical 1): return
    None and let the caller emit an explicit "not scorable" row instead of
    propagating the exception.
    """
    classes = np.unique(labels)
    if len(classes) < 2:
        base_rate = float(np.mean(labels)) if len(labels) else float("nan")
        logger.warning(
            f"{cell_name}: single-class labels (base rate {base_rate:.3f}) — not scorable"
        )
        return None
    return fit_calibration(precip_mm, labels)


def _temporal_split(
    sub: pd.DataFrame, fit_frac: float = FIT_SPLIT_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by issuance date: the earliest `fit_frac` of unique dates
    go to the fit half, the remaining held-out dates to the score half.

    A random row split would leak same-day, cross-station correlation
    across the fit/score boundary; for a time series the date is the
    correct split unit (Task 4 fix-round-1 Critical 2).
    """
    dates = np.sort(sub["issuance"].unique())
    if len(dates) < 2:
        empty = sub.iloc[0:0]
        return empty, empty
    cut_idx = int(len(dates) * fit_frac)
    cut_idx = min(max(cut_idx, 1), len(dates) - 1)
    cut = dates[cut_idx]
    fit_df = sub[sub["issuance"] < cut]
    score_df = sub[sub["issuance"] >= cut]
    return fit_df, score_df


def _score_cell(view: pd.DataFrame, label_col: str, cell_name: str) -> dict:
    """Fit calibration on the earliest 70% of issuance dates, score
    BSS-vs-climatology on the held-out latest 30%, and report both.

    Fitting and scoring on the same in-sample rows lets a 2-parameter
    logistic essentially always beat the in-sample base rate — a simulated
    zero-information predictor passed 20/20 trials under that scheme (Task 4
    fix-round-1 Critical 2). Anything that gates a design decision must use
    the out-of-sample number.
    """
    sub = view[np.isfinite(view[label_col].astype(float))]
    n = int(len(sub))
    if n < MIN_CELL_N:
        return {"status": "insufficient", "n": n}

    fit_df, score_df = _temporal_split(sub)
    if len(fit_df) < MIN_SPLIT_N or len(score_df) < MIN_SPLIT_N:
        return {
            "status": "insufficient_split",
            "n": n,
            "n_fit": int(len(fit_df)),
            "n_score": int(len(score_df)),
        }

    y_fit = fit_df[label_col].to_numpy(dtype=np.float32)
    y_score = score_df[label_col].to_numpy(dtype=np.float32)

    fit = _safe_fit_calibration(fit_df["precip_mm"].to_numpy(), y_fit, cell_name)
    if fit is None:
        return {"status": "single_class", "n": n, "base_rate": float(sub[label_col].mean())}
    a, b = fit

    p_fit = apply_calibration(fit_df["precip_mm"].to_numpy(), a, b)
    bss_in = float(brier_skill_score(p_fit.reshape(-1, 1), y_fit.reshape(-1, 1))[0])
    p_score = apply_calibration(score_df["precip_mm"].to_numpy(), a, b)
    bss_out = float(brier_skill_score(p_score.reshape(-1, 1), y_score.reshape(-1, 1))[0])

    # Refit on the full fine-tune window for seeding the residual head's
    # prior. These a/b are NOT what produced bss_in/bss_out above.
    full_fit = _safe_fit_calibration(
        sub["precip_mm"].to_numpy(),
        sub[label_col].to_numpy(dtype=np.float32),
        f"{cell_name} (full window)",
    )
    a_full, b_full = full_fit if full_fit is not None else (a, b)

    return {
        "status": "ok",
        "n": n,
        "n_fit": int(len(fit_df)),
        "n_score": int(len(score_df)),
        "base_rate": float(sub[label_col].mean()),
        # Base rate of the HELD-OUT half only. A degenerate score-half (e.g.
        # single-class) makes brier_skill_score's bs_clim>1e-9 guard return
        # a flat 0.0 that looks like an ordinary "no skill" result — this
        # column is what lets a reader spot that degeneracy instead of
        # mistaking it for a real measurement (fix-round-2 NEW-5).
        "score_base_rate": float(np.mean(y_score)),
        "bss_in_sample": bss_in,
        "bss_out_of_sample": bss_out,
        "calibration_a": a_full,
        "calibration_b": b_full,
    }


def _build_matched(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Inner-join gfs and graphcast forecasts for one horizon on
    (valid_time, station, issuance) — the labels are identical for both
    models at a given (valid_time, station), so joining on them too yields a
    single unsuffixed copy.

    Every downstream comparison (per-model BSS tables and the gate) scores
    both models on identically the same rows, so a coverage gap in one model
    cannot inflate or deflate its BSS relative to the other (Task 4
    fix-round-1 Minors: "gate takes max() across models ... different row
    sets"). This is also why the gate's coverage check must be based on
    ``len(matched)`` / the scored cell's ``n``, not either model's own raw
    fetch count (fix-round-2 NEW-1): a gap in one model shrinks this
    intersection for BOTH models even though the unaffected model's own
    fetch coverage still reads 100%.
    """
    cols = [
        "valid_time", "station", "issuance", "precip_mm",
        "era5_rain", "metar_rain", "metar_valid",
    ]
    g = df[(df.model == "gfs") & (df.horizon_h == h)][cols]
    c = df[(df.model == "graphcast") & (df.horizon_h == h)][cols]
    return g.merge(
        c,
        on=["valid_time", "station", "issuance", "era5_rain", "metar_rain", "metar_valid"],
        suffixes=("_g", "_c"),
    )


def _model_view(matched: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Extract one model's rows from a matched gfs/graphcast frame, keeping
    only the (valid_time, station) pairs both models actually cover."""
    return pd.DataFrame(
        {
            "issuance": matched["issuance"],
            "precip_mm": matched[f"precip_mm_{suffix}"],
            "era5_rain": matched["era5_rain"],
            "metar_rain": matched["metar_rain"],
            "metar_valid": matched["metar_valid"],
        }
    )


def _score_pair(matched: pd.DataFrame, label_col: str, cell_name: str) -> dict:
    """Same fit/score temporal-split discipline as _score_cell, applied to
    the reference-forecast comparison (GraphCast-GFS vs raw GFS)."""
    sub = matched[np.isfinite(matched[label_col].astype(float))]
    n = int(len(sub))
    if n < MIN_CELL_N:
        return {"status": "insufficient", "n": n}

    fit_df, score_df = _temporal_split(sub)
    if len(fit_df) < MIN_SPLIT_N or len(score_df) < MIN_SPLIT_N:
        return {
            "status": "insufficient_split",
            "n": n,
            "n_fit": int(len(fit_df)),
            "n_score": int(len(score_df)),
        }

    y_fit = fit_df[label_col].to_numpy(dtype=np.float32)
    y_score = score_df[label_col].to_numpy(dtype=np.float32)

    fit_g = _safe_fit_calibration(fit_df["precip_mm_g"].to_numpy(), y_fit, f"{cell_name} gfs")
    fit_c = _safe_fit_calibration(
        fit_df["precip_mm_c"].to_numpy(), y_fit, f"{cell_name} graphcast"
    )
    if fit_g is None or fit_c is None:
        return {"status": "single_class", "n": n, "base_rate": float(sub[label_col].mean())}
    ag, bg = fit_g
    ac, bc = fit_c

    def _rel(sub_df: pd.DataFrame, y: np.ndarray) -> float:
        pg = apply_calibration(sub_df["precip_mm_g"].to_numpy(), ag, bg).reshape(-1, 1)
        pc = apply_calibration(sub_df["precip_mm_c"].to_numpy(), ac, bc).reshape(-1, 1)
        return float(brier_skill_score_vs_reference(pc, pg, y.reshape(-1, 1))[0])

    return {
        "status": "ok",
        "n": n,
        "n_fit": int(len(fit_df)),
        "n_score": int(len(score_df)),
        "score_base_rate": float(np.mean(y_score)),
        "bss_in_sample": _rel(fit_df, y_fit),
        "bss_out_of_sample": _rel(score_df, y_score),
    }


def _emit_cell_row(model: str, h: int, cell: dict) -> str:
    if cell["status"] == "insufficient":
        return (
            f"| {model} | {h}h | {cell['n']} | — | — | — | — | "
            f"insufficient (n<{MIN_CELL_N}) |"
        )
    if cell["status"] == "insufficient_split":
        return (
            f"| {model} | {h}h | {cell['n']} | — | — | — | — | insufficient split "
            f"(fit={cell['n_fit']}, score={cell['n_score']}) |"
        )
    if cell["status"] == "single_class":
        return (
            f"| {model} | {h}h | {cell['n']} | {cell['base_rate']:.3f} | — | — | — | "
            "single-class — not scorable |"
        )
    # "ok" means scorable, NOT skilful — a negative out-of-sample BSS is a
    # perfectly valid "ok" result that says the forecast has no measurable
    # skill here. Show that explicitly so a skim of this column can't
    # mistake a negative result for a pass (fix-round-2 presentation fix).
    verdict = "skilful (BSS>0)" if cell["bss_out_of_sample"] > 0 else "not skilful (BSS<=0)"
    return (
        f"| {model} | {h}h | {cell['n']:,} (fit {cell['n_fit']:,}/"
        f"score {cell['n_score']:,}) | {cell['base_rate']:.3f} | "
        f"{cell['score_base_rate']:.3f} | "
        f"{cell['bss_in_sample']:+.4f} | {cell['bss_out_of_sample']:+.4f} | {verdict} |"
    )


def _write_empty_window_report(
    out: Path, args: argparse.Namespace, n_days: int, fetch_stats: dict
) -> None:
    """Write a minimal report.md/gate.json and return cleanly when no
    forecast rows exist at all for the requested window, instead of letting
    `fc.model` (or anything downstream) raise on a columnless/empty frame
    (fix-round-2 NEW-4).
    """
    incomplete = fetch_stats["incomplete_days"]
    shown = incomplete[:20]
    more = len(incomplete) - len(shown)
    lines = [
        "# Phase 0 — Forecast Benchmark",
        "",
        f"Window: {args.start} to {args.end}  ({n_days} days)",
        "",
        "**No forecast rows were fetched for this window** — every "
        "requested day is incomplete (0 complete shards). Nothing to score.",
        "",
        f"Incomplete days ({len(incomplete)}): " + ", ".join(shown)
        + (f", +{more} more" if more > 0 else ""),
        (
            f"- GFS fetch exceptions: {fetch_stats['gfs']['fetch_exceptions']:,} of "
            f"{fetch_stats['gfs']['attempted']:,} attempted"
        ),
        (
            "- GraphCast-GFS fetch exceptions: "
            f"{fetch_stats['graphcast']['fetch_exceptions']:,} of "
            f"{fetch_stats['graphcast']['attempted']:,} attempted"
        ),
        "",
        "## Gate",
        "",
        "- Scored coverage at 48 h: 0.0%",
        f"- Thresholds: out-of-sample BSS > {GATE_MIN_BSS_48H}, "
        f"coverage >= {GATE_MIN_COVERAGE:.0%}",
        "- **FAIL — revisit design**",
    ]
    (out / "report.md").write_text("\n".join(lines))
    (out / "gate.json").write_text(
        json.dumps(
            {
                "passed": False,
                "best_model_48h": None,
                "scored_coverage_48h_best_model": 0.0,
                "fetch_coverage_48h_best_model": 0.0,
                "fetch_stats": fetch_stats,
                "results": {},
            },
            indent=2,
            default=str,
        )
    )
    logger.error("no complete forecast shards for the requested window — nothing to score")
    logger.info(f"wrote {out/'report.md'} and {out/'gate.json'}")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="era5_recent.parquet")
    ap.add_argument("--metar", required=True, help="metar_thai.parquet")
    ap.add_argument("--out", default="data/phase0")
    ap.add_argument("--start", default=str(GRAPHCAST_IDX_FROM.date()))
    ap.add_argument("--end", default=str(FIT_END.date()))
    ap.add_argument(
        "--allow-outside-fit-window",
        action="store_true",
        help="Allow --start/--end outside the fine-tune window. The "
        "calibration fit must never touch validation or test data, so this "
        "is an explicit, deliberate override — not a default escape hatch.",
    )
    args = ap.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    if not args.allow_outside_fit_window and (start_ts < FIT_START or end_ts > FIT_END):
        raise SystemExit(
            f"--start/--end ({args.start}..{args.end}) falls outside the "
            f"fine-tune window ({FIT_START.date()}..{FIT_END.date()}). "
            "Fitting calibration outside this window risks touching "
            "validation/test data. Pass --allow-outside-fit-window to "
            "override deliberately."
        )

    # Fail in seconds, not hours: a typo'd --labels/--metar path would
    # otherwise only surface after the multi-hour fetch loop, in
    # build_labels(), wasting the entire run.
    labels_path = Path(args.labels)
    metar_path = Path(args.metar)
    missing = [str(p) for p in (labels_path, metar_path) if not p.exists()]
    if missing:
        raise SystemExit(f"input file(s) not found: {', '.join(missing)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"

    fetch_forecasts(start_ts, end_ts, shard_dir)
    fetch_stats = _aggregate_day_stats(shard_dir, start_ts, end_ts)
    n_days = len(pd.date_range(start_ts, end_ts, freq="D", tz="UTC"))
    expected_per_cell = n_days * len(STATION_COORDS)

    fc = _load_shards(shard_dir, start_ts, end_ts)
    fc.to_parquet(out / "forecasts.parquet", index=False)
    logger.info(f"forecast rows: {len(fc):,}")

    if fc.empty or "model" not in fc.columns:
        _write_empty_window_report(out, args, n_days, fetch_stats)
        return

    lbl, label_diag = build_labels(labels_path, metar_path, fc["valid_time"])
    df = fc.merge(lbl, on=["valid_time", "station"], how="inner")
    matched_by_h = {h: _build_matched(df, h) for h in HORIZONS_H}

    # Raw per-model fetch coverage — diagnostic only. This is NOT what the
    # gate uses (fix-round-2 NEW-1): see the "Scored coverage" section below.
    coverage: dict[tuple[str, int], float] = {}
    for model in ("gfs", "graphcast"):
        for h in HORIZONS_H:
            fetched = int(((fc.model == model) & (fc.horizon_h == h)).sum())
            coverage[(model, h)] = fetched / expected_per_cell if expected_per_cell else 0.0

    lines = ["# Phase 0 — Forecast Benchmark", ""]
    lines.append(f"Window: {args.start} to {args.end}  ({n_days} days)")
    lines.append(
        'Horizon definitions: "24h" scores the 6-hour accumulation window '
        "ending at issuance+24h, forecast from the run published 30h "
        "earlier (the freshest run publishable at a 00Z issuance is the "
        'previous day\'s 18Z). "48h" scores the window ending at '
        "issuance+48h, from the run published 54h earlier. Both are "
        "genuine >=24h / >=48h forecasts."
    )
    lines.append("")
    lines.append(f"Forecast rows: {len(fc):,}  |  matched to labels: {len(df):,}")
    lines.append(
        f"ERA5 label windows: {label_diag['era5_windows_total']:,} total, "
        f"{label_diag['era5_windows_dropped_incomplete']:,} dropped "
        "(fewer than all 6 hours present in the accumulation window — a "
        "partial sum would silently under-count rain)."
    )
    lines.append(
        f"ERA5 raw null precipitation rows in the needed window: "
        f"{label_diag['era5_raw_null_precip_rows']:,} (checked on the raw "
        "frame before label construction — a null-but-present row would "
        "otherwise be silently filled to 0.0 mm and reported as complete; "
        "a nonzero count here means the run FAILED LOUDLY earlier and this "
        "line would not exist, so 0 is the only value that can ever be "
        "printed here)."
    )
    lines.append("")

    lines.append("## Fetch coverage (raw, per model — diagnostic only)")
    lines.append("")
    lines.append(
        "This is each model's own fetch success rate. It is NOT what the "
        "gate checks — see 'Scored coverage' below, which reflects the "
        "gfs∩graphcast matched row set the BSS tables actually score "
        "against. A model can read 100% here while its scored coverage is "
        "much lower, if the OTHER model has gaps (fix-round-2 NEW-1)."
    )
    lines.append("")
    lines.append("| model | horizon | fetched | expected | fetch coverage |")
    lines.append("|---|---|---|---|---|")
    for model in ("gfs", "graphcast"):
        for h in HORIZONS_H:
            fetched = int(((fc.model == model) & (fc.horizon_h == h)).sum())
            cov = coverage[(model, h)]
            lines.append(
                f"| {model} | {h}h | {fetched:,} | {expected_per_cell:,} | {cov:.1%} |"
            )
    incomplete = fetch_stats["incomplete_days"]
    if incomplete:
        shown = incomplete[:15]
        more = len(incomplete) - len(shown)
        lines.append(
            f"- Incomplete days (no shard written yet, will retry next run): "
            f"{len(incomplete):,} — " + ", ".join(shown)
            + (f", +{more} more" if more > 0 else "")
        )
    else:
        lines.append("- Incomplete days: 0")
    lines.append(
        f"- Leakage/bucket-end skips (both models, both horizons): "
        f"{fetch_stats['run_select_skips']:,}"
    )
    lines.append(
        f"- GFS fetch exceptions: {fetch_stats['gfs']['fetch_exceptions']:,} of "
        f"{fetch_stats['gfs']['attempted']:,} attempted"
    )
    lines.append(
        "- GraphCast-GFS fetch exceptions: "
        f"{fetch_stats['graphcast']['fetch_exceptions']:,} of "
        f"{fetch_stats['graphcast']['attempted']:,} attempted"
    )
    lines.append("")

    results: dict = {}

    lines.append("## Scored against `era5_rain` (>= 1.0mm accumulated over the window)")
    lines.append("")
    lines.append(
        "Fit on the earliest 70% of days in the requested window by "
        "issuance date, scored on the held-out latest 30% — never the same "
        "rows. `calibration_a`/`calibration_b` in gate.json are refit on "
        "the full window once scoring is complete, for seeding the "
        "residual head's prior; they are NOT what produced the BSS numbers "
        "below. `score-half base rate` is the held-out half's own base "
        "rate — if it's 0.000 or 1.000 the cell is degenerate even though "
        "the verdict column still reads normally (see gate.json)."
    )
    lines.append("")
    lines.append(
        "| model | horizon | n | base rate | score-half base rate | "
        "BSS in-sample | BSS out-of-sample | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for model in ("gfs", "graphcast"):
        for h in HORIZONS_H:
            view = _model_view(matched_by_h[h], "g" if model == "gfs" else "c")
            cell = _score_cell(view, "era5_rain", f"{model}_{h}h_era5_rain")
            results[f"{model}_{h}h_era5_rain"] = cell
            lines.append(_emit_cell_row(model, h, cell))
    lines.append("")

    lines.append("## Scored against `metar_rain` (any rain_event observed in the window)")
    lines.append("")
    lines.append(
        "**Not the same event as `era5_rain`.** `era5_rain` is a >=1.0mm "
        "accumulated-depth threshold; `metar_rain` is occurrence-only (was "
        "any rain reported at all, from present-weather codes RA/TS/SH), "
        "with no intensity information — the Thai ASOS feed's precip_mm "
        "field is identically zero across the whole archive upstream and "
        "is not usable. Treat this table as a secondary observational "
        "sanity check, not a like-for-like comparison with the ERA5 table. "
        "It is NOT a gate input."
    )
    lines.append("")
    lines.append(
        "| model | horizon | n | base rate | score-half base rate | "
        "BSS in-sample | BSS out-of-sample | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for model in ("gfs", "graphcast"):
        for h in HORIZONS_H:
            view = _model_view(matched_by_h[h], "g" if model == "gfs" else "c")
            view = view[view["metar_valid"]]
            cell = _score_cell(view, "metar_rain", f"{model}_{h}h_metar_rain")
            results[f"{model}_{h}h_metar_rain"] = cell
            lines.append(_emit_cell_row(model, h, cell))
    lines.append("")

    lines.append("## GraphCast-GFS vs raw GFS (the bar)")
    lines.append("")
    lines.append(
        "Positive = GraphCast-GFS beats raw GFS at the same events (scored "
        "against `era5_rain`, same fit/score split as above)."
    )
    lines.append("")
    lines.append(
        "| horizon | n (fit/score) | score-half base rate | BSS in-sample | "
        "BSS out-of-sample | verdict |"
    )
    lines.append("|---|---|---|---|---|---|")
    for h in HORIZONS_H:
        cell = _score_pair(matched_by_h[h], "era5_rain", f"graphcast_vs_gfs_{h}h")
        results[f"graphcast_vs_gfs_{h}h"] = cell
        if cell["status"] == "ok":
            verdict = (
                "graphcast better (BSS>0)"
                if cell["bss_out_of_sample"] > 0
                else "graphcast not better (BSS<=0)"
            )
            lines.append(
                f"| {h}h | {cell['n_fit']:,}/{cell['n_score']:,} | "
                f"{cell['score_base_rate']:.3f} | {cell['bss_in_sample']:+.4f} | "
                f"{cell['bss_out_of_sample']:+.4f} | {verdict} |"
            )
        else:
            lines.append(f"| {h}h | {cell.get('n', 0)} | — | — | — | {cell['status']} |")
    lines.append("")

    # Scored coverage — this is what the gate uses (fix-round-2 NEW-1). The
    # raw per-model "Fetch coverage" table above can hide a coverage gap: if
    # graphcast fails on 40% of days while gfs fetches cleanly throughout,
    # gfs's own raw fetch coverage still reads 100% even though the
    # gfs∩graphcast matched set — what era5_rain is actually scored against
    # — has shrunk by the same 40%. Gate on THIS number instead.
    lines.append("## Scored coverage (this is what the gate uses)")
    lines.append("")
    lines.append(
        "Computed from the SAME gfs∩graphcast matched row set the BSS "
        "tables above score against, not from either model's own raw fetch "
        "count. A coverage gap in one model reduces both models' scored "
        "coverage here, even when the unaffected model's raw fetch "
        "coverage above still reads 100%."
    )
    lines.append("")
    lines.append("| horizon | matched (gfs∩graphcast) | expected | matched coverage |")
    lines.append("|---|---|---|---|")
    for h in HORIZONS_H:
        matched_n = len(matched_by_h[h])
        matched_cov = matched_n / expected_per_cell if expected_per_cell else 0.0
        lines.append(f"| {h}h | {matched_n:,} | {expected_per_cell:,} | {matched_cov:.1%} |")
    lines.append("")
    lines.append("| model | horizon | scored n (era5_rain) | expected | scored coverage |")
    lines.append("|---|---|---|---|---|")
    scored_coverage: dict[tuple[str, int], float] = {}
    for model in ("gfs", "graphcast"):
        for h in HORIZONS_H:
            cell_n = results.get(f"{model}_{h}h_era5_rain", {}).get("n", 0)
            cov = cell_n / expected_per_cell if expected_per_cell else 0.0
            scored_coverage[(model, h)] = cov
            lines.append(f"| {model} | {h}h | {cell_n:,} | {expected_per_cell:,} | {cov:.1%} |")
    lines.append("")

    # Gate — keys off era5_rain OUT-OF-SAMPLE BSS only, at 48h, from the SAME
    # matched (gfs ∩ graphcast) row set the head-to-head table uses, so a
    # coverage gap in one model cannot inflate its BSS relative to the other.
    # Also requires >=90% SCORED coverage at 48h for the winning model — not
    # raw fetch coverage (fix-round-2 NEW-1). metar_rain is never a gate
    # input.
    gfs_48 = results.get("gfs_48h_era5_rain", {"status": "insufficient", "n": 0})
    gc_48 = results.get("graphcast_48h_era5_rain", {"status": "insufficient", "n": 0})
    scorable = [
        (name, cell) for name, cell in (("gfs", gfs_48), ("graphcast", gc_48))
        if cell["status"] == "ok"
    ]

    lines.append("## Gate")
    lines.append("")
    best_model: str | None = None
    best_oos: float | None = None
    best_in: float | None = None
    cov = 0.0
    raw_cov = 0.0
    if not scorable:
        lines.append("- No model was scorable at 48 h against `era5_rain` (see table above).")
        passed = False
    else:
        best_model, best_cell = max(scorable, key=lambda t: t[1]["bss_out_of_sample"])
        best_oos = best_cell["bss_out_of_sample"]
        best_in = best_cell["bss_in_sample"]
        cov = scored_coverage.get((best_model, 48), 0.0)
        raw_cov = coverage.get((best_model, 48), 0.0)
        lines.append(
            f"- Best model at 48 h (by out-of-sample BSS vs climatology): **{best_model}**"
        )
        lines.append(
            f"- Out-of-sample BSS at 48 h: **{best_oos:+.4f}** (in-sample: {best_in:+.4f})"
        )
        lines.append(
            f"- Scored coverage at 48 h for {best_model}: **{cov:.1%}** "
            f"(raw fetch coverage: {raw_cov:.1%})"
        )
        passed = best_oos > GATE_MIN_BSS_48H and cov >= GATE_MIN_COVERAGE
    lines.append(
        f"- Thresholds: out-of-sample BSS > {GATE_MIN_BSS_48H}, "
        f"scored coverage >= {GATE_MIN_COVERAGE:.0%}"
    )
    lines.append(f"- **{'PASS — proceed' if passed else 'FAIL — revisit design'}**")
    if not passed:
        lines.append("")
        lines.append(
            "The prior does not clear both the skill and scored-coverage "
            "bars at 48 h. Do NOT commission the grid download. Report and "
            "revisit."
        )

    (out / "report.md").write_text("\n".join(lines))
    (out / "gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "best_model_48h": best_model,
                "best_bss_48h_out_of_sample": best_oos,
                "best_bss_48h_in_sample": best_in,
                "scored_coverage_48h_best_model": cov,
                "fetch_coverage_48h_best_model": raw_cov,
                "fetch_stats": fetch_stats,
                "fetch_coverage": {f"{m}_{h}h": c for (m, h), c in coverage.items()},
                "scored_coverage": {f"{m}_{h}h": c for (m, h), c in scored_coverage.items()},
                "label_diagnostics": label_diag,
                "results": results,
            },
            indent=2,
            default=str,
        )
    )
    logger.info(f"wrote {out/'report.md'} and {out/'gate.json'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
