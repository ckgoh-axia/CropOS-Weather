#!/usr/bin/env python3
"""Phase 0 — measure the bar before building anything.

Scores the free public rain forecasts (raw GFS and operational GraphCast-GFS)
at the 16 Thai METAR stations at 24 h and 48 h lead. That score is the bar the
redesigned model must beat, and it gates the grid download.

Every forecast is fetched through select_run(), so the publication-latency
rule in spec 4.3 is enforced for the benchmark exactly as it will be at
inference. A benchmark that cheats would set the bar too high.

Forecasts are checkpointed as one parquet shard per day under
``<out>/shards/`` as soon as that day completes, so an interrupted multi-hour
run resumes from the next unfetched day instead of restarting.
``<out>/forecasts.parquet`` is always rebuilt from exactly the shards for the
requested ``--start``/``--end``, so it can never silently reflect a different
window than the one just requested.

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

# GraphCast-GFS archive begins here; the fine-tune window starts with it.
FIT_START = pd.Timestamp("2024-02-05", tz="UTC")
FIT_END = pd.Timestamp("2025-06-30", tz="UTC")

# Gate: the best calibrated prior must clear a small margin above sampling
# noise on OUT-OF-SAMPLE BSS (never in-sample — a 2-parameter logistic fit
# and scored on the same rows essentially always beats the in-sample base
# rate; see Task 4 fix-round-1 Critical 2) and must be fetched at >=90%
# coverage (Important 3).
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


def fetch_forecasts(start: pd.Timestamp, end: pd.Timestamp, shard_dir: Path) -> dict:
    """Fetch GFS and GraphCast-GFS accumulated precip for each day and horizon.

    Writes one parquet shard per day to ``shard_dir`` as soon as that day
    completes. A day whose shard already exists is skipped without any
    network call, so re-running after an interruption resumes rather than
    restarts (Important 2).

    Returns fetch diagnostics: attempted/exception counts per model, the
    count of (day, horizon) pairs skipped by the leakage guard or the 6h
    bucket-end check, and the number of days processed.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "days": 0,
        "run_select_skips": 0,
        "gfs": {"attempted": 0, "fetch_exceptions": 0},
        "graphcast": {"attempted": 0, "fetch_exceptions": 0},
    }
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        stats["days"] += 1
        shard_path = shard_dir / f"{day.date()}.parquet"
        if shard_path.exists():
            logger.info(f"{day.date()}: shard exists, skipping fetch")
            continue

        rows: list[dict] = []
        issuance = day  # 00Z issuance
        for h in HORIZONS_H:
            valid = issuance + pd.Timedelta(hours=h)
            try:
                run, lead = select_run(issuance, valid)
            except ValueError as exc:
                logger.warning(f"skip {issuance} h={h}: {exc}")
                stats["run_select_skips"] += 1
                continue
            if lead - bucket_start(lead) != ACCUM_WINDOW_H:
                # select_run may return a lead that is not a 6h bucket end.
                logger.debug(f"skip {issuance} h={h}: lead {lead} not a 6h bucket end")
                stats["run_select_skips"] += 1
                continue
            pattern = _apcp_pattern(lead)
            for model, bucket, keyfn in (
                ("gfs", BUCKET_GFS, gfs_key),
                ("graphcast", BUCKET_GRAPHCAST, graphcast_key),
            ):
                stats[model]["attempted"] += 1
                try:
                    vals = fetch_point_values(
                        bucket, keyfn(run, lead), pattern, STATION_COORDS
                    )
                except Exception as exc:  # noqa: BLE001 — log, count, continue
                    logger.warning(f"{model} {run} f{lead:03d}: {exc}")
                    stats[model]["fetch_exceptions"] += 1
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
        day_df = pd.DataFrame(rows)
        _write_shard(shard_path, day_df)
        logger.info(f"{day.date()}: {len(day_df):,} rows written to {shard_path.name}")
    return stats


def _load_shards(shard_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Concatenate the per-day shards covering [start, end].

    forecasts.parquet is always rebuilt from these shards for exactly the
    requested window, so a stale cache from a different --start/--end can
    never be silently reused (Important 2). fetch_forecasts writes a shard
    for every day it processes (even an all-failed day gets an empty
    shard), so a missing shard here means fetch_forecasts did not run to
    completion for this window — fail loudly rather than silently drop days.
    """
    cols = [
        "issuance", "valid_time", "horizon_h", "model", "run", "lead_h",
        "station", "precip_mm",
    ]
    frames = []
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        p = shard_dir / f"{day.date()}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"missing shard for {day.date()} in {shard_dir} — a previous "
                "run may have been killed mid-day. Re-run to fetch it."
            )
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def _read_era5_window(path: Path, needed_ts: set[pd.Timestamp]) -> pd.DataFrame:
    """Read only the exact ERA5 timestamps the label windows need.

    era5_recent.parquet is ~60.7M rows. A 17-month date-range filter alone
    still reads ~24M rows though only 6 of every 24 hours (the accumulation
    window is always 18:00-23:00 UTC of the day before a 00Z-aligned valid
    time) are ever used. Filtering to the exact needed timestamps instead of
    a range narrows this to what the label builder actually touches
    (Important 4). The filter is strictly narrowing — it cannot change label
    values.
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

    diagnostics: {"era5_windows_total", "era5_windows_dropped_incomplete"}

    Note: METAR's precip_mm is deliberately not used. The Thai ASOS feed has
    no p01i column, so upstream (src/ingestion/metar.py) precip_mm is filled
    with the pd.to_numeric(..., 0) default and is identically zero across
    the entire archive — verified independently (1,034,434 rows, min/max/sum
    all 0.0). Fitting a logistic calibration against it raises immediately
    on a single-class label and previously crashed a multi-hour run with
    nothing written (Task 4 fix-round-1 Critical 1).
    """
    from src.features.dataset import _build_era5_label_df

    offsets = [pd.Timedelta(hours=k) for k in range(ACCUM_WINDOW_H)]
    unique_vt = sorted(set(valid_times))
    needed_ts = {vt - o for vt in unique_vt for o in offsets}

    era5 = _read_era5_window(era5_path, needed_ts)
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
    sets").
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


def _score_pair(
    matched: pd.DataFrame, label_col: str, mask_col: str | None, cell_name: str
) -> dict:
    """Same fit/score temporal-split discipline as _score_cell, applied to
    the reference-forecast comparison (GraphCast-GFS vs raw GFS)."""
    view = matched[matched[mask_col]] if mask_col else matched
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
        "bss_in_sample": _rel(fit_df, y_fit),
        "bss_out_of_sample": _rel(score_df, y_score),
    }


def _emit_cell_row(model: str, h: int, cell: dict) -> str:
    if cell["status"] == "insufficient":
        return f"| {model} | {h}h | {cell['n']} | — | — | — | insufficient (n<{MIN_CELL_N}) |"
    if cell["status"] == "insufficient_split":
        return (
            f"| {model} | {h}h | {cell['n']} | — | — | — | insufficient split "
            f"(fit={cell['n_fit']}, score={cell['n_score']}) |"
        )
    if cell["status"] == "single_class":
        return (
            f"| {model} | {h}h | {cell['n']} | {cell['base_rate']:.3f} | — | — | "
            "single-class — not scorable |"
        )
    return (
        f"| {model} | {h}h | {cell['n']:,} (fit {cell['n_fit']:,}/"
        f"score {cell['n_score']:,}) | {cell['base_rate']:.3f} | "
        f"{cell['bss_in_sample']:+.4f} | {cell['bss_out_of_sample']:+.4f} | ok |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="era5_recent.parquet")
    ap.add_argument("--metar", required=True, help="metar_thai.parquet")
    ap.add_argument("--out", default="data/phase0")
    ap.add_argument("--start", default=str(FIT_START.date()))
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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"

    fetch_stats = fetch_forecasts(start_ts, end_ts, shard_dir)
    fc = _load_shards(shard_dir, start_ts, end_ts)
    fc.to_parquet(out / "forecasts.parquet", index=False)
    logger.info(f"forecast rows: {len(fc):,}")

    lbl, label_diag = build_labels(Path(args.labels), Path(args.metar), fc["valid_time"])
    df = fc.merge(lbl, on=["valid_time", "station"], how="inner")
    matched_by_h = {h: _build_matched(df, h) for h in HORIZONS_H}

    n_days = fetch_stats["days"]
    expected_per_cell = n_days * len(STATION_COORDS)
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
    lines.append("")

    lines.append("## Fetch coverage")
    lines.append("")
    lines.append("| model | horizon | fetched | expected | coverage |")
    lines.append("|---|---|---|---|---|")
    for model in ("gfs", "graphcast"):
        for h in HORIZONS_H:
            fetched = int(((fc.model == model) & (fc.horizon_h == h)).sum())
            cov = coverage[(model, h)]
            lines.append(
                f"| {model} | {h}h | {fetched:,} | {expected_per_cell:,} | {cov:.1%} |"
            )
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
        "below."
    )
    lines.append("")
    lines.append(
        "| model | horizon | n | base rate | BSS in-sample | BSS out-of-sample | status |"
    )
    lines.append("|---|---|---|---|---|---|---|")
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
        "| model | horizon | n | base rate | BSS in-sample | BSS out-of-sample | status |"
    )
    lines.append("|---|---|---|---|---|---|---|")
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
    lines.append("| horizon | n (fit/score) | BSS in-sample | BSS out-of-sample | status |")
    lines.append("|---|---|---|---|---|")
    for h in HORIZONS_H:
        cell = _score_pair(matched_by_h[h], "era5_rain", None, f"graphcast_vs_gfs_{h}h")
        results[f"graphcast_vs_gfs_{h}h"] = cell
        if cell["status"] == "ok":
            lines.append(
                f"| {h}h | {cell['n_fit']:,}/{cell['n_score']:,} | "
                f"{cell['bss_in_sample']:+.4f} | {cell['bss_out_of_sample']:+.4f} | ok |"
            )
        else:
            lines.append(f"| {h}h | {cell.get('n', 0)} | — | — | {cell['status']} |")
    lines.append("")

    # Gate — keys off era5_rain OUT-OF-SAMPLE BSS only, at 48h, from the SAME
    # matched (gfs ∩ graphcast) row set the head-to-head table uses, so a
    # coverage gap in one model cannot inflate its BSS relative to the other.
    # Also requires >=90% fetch coverage at 48h for the winning model.
    # metar_rain is never a gate input.
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
    if not scorable:
        lines.append("- No model was scorable at 48 h against `era5_rain` (see table above).")
        passed = False
    else:
        best_model, best_cell = max(scorable, key=lambda t: t[1]["bss_out_of_sample"])
        best_oos = best_cell["bss_out_of_sample"]
        best_in = best_cell["bss_in_sample"]
        cov = coverage.get((best_model, 48), 0.0)
        lines.append(
            f"- Best model at 48 h (by out-of-sample BSS vs climatology): **{best_model}**"
        )
        lines.append(
            f"- Out-of-sample BSS at 48 h: **{best_oos:+.4f}** (in-sample: {best_in:+.4f})"
        )
        lines.append(f"- Fetch coverage at 48 h for {best_model}: {cov:.1%}")
        passed = best_oos > GATE_MIN_BSS_48H and cov >= GATE_MIN_COVERAGE
    lines.append(
        f"- Thresholds: out-of-sample BSS > {GATE_MIN_BSS_48H}, "
        f"coverage >= {GATE_MIN_COVERAGE:.0%}"
    )
    lines.append(f"- **{'PASS — proceed' if passed else 'FAIL — revisit design'}**")
    if not passed:
        lines.append("")
        lines.append(
            "The prior does not clear both the skill and coverage bars at "
            "48 h. Do NOT commission the grid download. Report and revisit."
        )

    (out / "report.md").write_text("\n".join(lines))
    (out / "gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "best_model_48h": best_model,
                "best_bss_48h_out_of_sample": best_oos,
                "best_bss_48h_in_sample": best_in,
                "coverage_48h_best_model": cov,
                "fetch_stats": fetch_stats,
                "coverage": {f"{m}_{h}h": c for (m, h), c in coverage.items()},
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
